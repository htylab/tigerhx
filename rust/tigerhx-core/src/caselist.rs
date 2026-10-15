//! The original GUI's case-list CSV (Filename, Apex, mat_in_inputdir,
//! nii_in_inputdir), read and written like tigerhx/core/io.py does.

use std::path::{Path, PathBuf};

use crate::error::Result;
use crate::fsutil::{abspath, path_str, write_atomic};

pub const CASE_COLUMNS: [&str; 4] = ["Filename", "Apex", "mat_in_inputdir", "nii_in_inputdir"];

/// Bytes as text: UTF-8 (with or without BOM), else Big5 (cp950), GBK, Latin-1.
fn decode(raw: &[u8]) -> String {
    let raw = raw.strip_prefix(b"\xef\xbb\xbf").unwrap_or(raw);
    if let Ok(s) = std::str::from_utf8(raw) {
        return s.to_string();
    }
    for enc in [encoding_rs::BIG5, encoding_rs::GBK] {
        let (s, had_errors) = enc.decode_without_bom_handling(raw);
        if !had_errors {
            return s.into_owned();
        }
    }
    raw.iter().map(|&b| b as char).collect()
}

/// Records of a CSV text (quotes and doubled quotes as Python's csv module).
fn parse(text: &str, delim: char) -> Vec<Vec<String>> {
    let mut rows = Vec::new();
    let (mut row, mut field) = (Vec::new(), String::new());
    let (mut quoted, mut chars) = (false, text.chars().peekable());
    let mut started = false;
    while let Some(c) = chars.next() {
        if quoted {
            if c == '"' {
                if chars.peek() == Some(&'"') {
                    field.push('"');
                    chars.next();
                } else {
                    quoted = false;
                }
            } else {
                field.push(c);
            }
            continue;
        }
        match c {
            '"' if field.is_empty() => {
                quoted = true;
                started = true;
            }
            '\r' | '\n' => {
                if c == '\r' && chars.peek() == Some(&'\n') {
                    chars.next();
                }
                if started || !field.is_empty() || !row.is_empty() {
                    row.push(std::mem::take(&mut field));
                    rows.push(std::mem::take(&mut row));
                }
                started = false;
            }
            c if c == delim => {
                row.push(std::mem::take(&mut field));
                started = true;
            }
            c => {
                field.push(c);
                started = true;
            }
        }
    }
    if started || !field.is_empty() || !row.is_empty() {
        row.push(field);
        rows.push(row);
    }
    rows
}

/// Rows as (column, value) maps, values stripped; ',' or ';' delimited
/// (whichever the header line has more of), as Excel writes them.
pub fn read_csv(path: &Path) -> Result<Vec<Vec<(String, String)>>> {
    let text = decode(&std::fs::read(path).map_err(|e| crate::Error::from(e).context(path.display()))?);
    let first = text.lines().next().unwrap_or("");
    let delim = if first.matches(';').count() > first.matches(',').count() { ';' } else { ',' };
    let mut rows = parse(&text, delim).into_iter();
    let Some(header) = rows.next() else { return Ok(Vec::new()) };
    let header: Vec<String> = header.iter().map(|h| h.trim().to_string()).collect();
    Ok(rows
        .map(|r| header.iter().enumerate().map(|(i, h)| (h.clone(), r.get(i).map(|v| v.trim().to_string()).unwrap_or_default())).collect())
        .collect())
}

/// One row of a case list.
#[derive(Clone, Debug, PartialEq)]
pub struct CaseRow {
    pub path: PathBuf,
    pub apex: Option<usize>,
    pub mat_in_inputdir: Option<bool>,
    pub nii_in_inputdir: Option<bool>,
}

/// Python's int(float(text)), None when it fails.
fn int_or_none(text: &str) -> Option<i64> {
    let f: f64 = text.trim().parse().ok()?;
    f.is_finite().then(|| f.trunc() as i64)
}

/// The rows of a case list: filename (relative names are taken from the
/// CSV's folder), apex (None when blank or negative) and the two output
/// flags (None when blank, so the global setting applies).
pub fn read_case_csv(path: &Path) -> Result<Vec<CaseRow>> {
    let base = abspath(path).parent().map(Path::to_path_buf).unwrap_or_default();
    let get = |r: &[(String, String)], k: &str| r.iter().find(|(h, _)| h == k).map(|(_, v)| v.clone()).unwrap_or_default();
    Ok(read_csv(path)?
        .into_iter()
        .filter_map(|r| {
            let name = get(&r, "Filename");
            let name = name.trim().trim_matches('"');
            if name.is_empty() {
                return None;
            }
            let flag = |k| int_or_none(&get(&r, k)).map(|v| v != 0);
            Some(CaseRow {
                path: abspath(&base.join(name)),
                apex: int_or_none(&get(&r, "Apex")).filter(|&a| a >= 0).map(|a| a as usize),
                mat_in_inputdir: flag("mat_in_inputdir"),
                nii_in_inputdir: flag("nii_in_inputdir"),
            })
        })
        .collect())
}

fn quote(field: &str) -> String {
    if field.contains([',', '"', '\r', '\n']) {
        format!("\"{}\"", field.replace('"', "\"\""))
    } else {
        field.to_string()
    }
}

/// UTF-8 with BOM (so Excel shows non-ASCII paths), CRLF line ends and
/// fields quoted only when needed, as Python's csv writer does.
pub fn write_csv(path: &Path, header: &[&str], rows: &[Vec<String>]) -> Result<()> {
    let mut text = String::from("\u{feff}");
    for r in std::iter::once(header.iter().map(|s| s.to_string()).collect::<Vec<_>>()).chain(rows.iter().cloned()) {
        text.push_str(&r.iter().map(|f| quote(f)).collect::<Vec<_>>().join(","));
        text.push_str("\r\n");
    }
    write_atomic(path, |f| {
        use std::io::Write;
        f.write_all(text.as_bytes())?;
        Ok(())
    })
}

/// A GenCSV row: the file, a blank apex and the two output flags.
pub fn gen_row(path: &Path, mat_in_inputdir: bool, nii_in_inputdir: bool) -> Vec<String> {
    vec![path_str(path), String::new(), (mat_in_inputdir as u8).to_string(), (nii_in_inputdir as u8).to_string()]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_quotes_and_delimiters() {
        assert_eq!(parse("a,\"b,c\",\"d\"\"e\"\r\n1,,3\n", ','), vec![vec!["a", "b,c", "d\"e"], vec!["1", "", "3"]]);
        assert_eq!(parse("x;y\n\n1;2", ';'), vec![vec!["x", "y"], vec!["1", "2"]]);
    }

    #[test]
    fn case_rows() {
        let dir = std::env::temp_dir().join(format!("tigerhx-csv-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("list.csv");
        write_csv(&p, &CASE_COLUMNS, &[
            vec!["a/x.nii.gz".into(), "4".into(), "0".into(), "1".into()],
            vec!["/abs/y, z.nii".into(), "".into(), "".into(), "".into()],
            vec!["w.nii".into(), "-1".into(), "2.0".into(), "x".into()],
        ])
        .unwrap();
        let bytes = std::fs::read(&p).unwrap();
        assert!(bytes.starts_with(b"\xef\xbb\xbfFilename,Apex,mat_in_inputdir,nii_in_inputdir\r\n"));
        let rows = read_case_csv(&p).unwrap();
        assert_eq!(rows[0], CaseRow { path: dir.join("a/x.nii.gz"), apex: Some(4), mat_in_inputdir: Some(false), nii_in_inputdir: Some(true) });
        assert_eq!(rows[1].path, PathBuf::from("/abs/y, z.nii"));
        assert_eq!((rows[1].apex, rows[1].mat_in_inputdir), (None, None));
        assert_eq!((rows[2].apex, rows[2].mat_in_inputdir, rows[2].nii_in_inputdir), (None, Some(true), None));
        std::fs::write(&p, b"Filename;Apex\r\nC:\\x\\\xa4\xa4.nii;3\r\n").unwrap(); // Big5 from Excel
        let rows = read_case_csv(&p).unwrap();
        assert_eq!(rows[0].apex, Some(3));
        assert!(path_str(&rows[0].path).contains('中'));
        std::fs::remove_dir_all(&dir).ok();
    }
}
