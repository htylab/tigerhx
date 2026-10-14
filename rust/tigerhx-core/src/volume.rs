//! A 4D array indexed (x, y, slice, frame), stored in NIfTI (Fortran) order:
//! x varies fastest, so one 2D slice of one frame is contiguous.

#[derive(Clone, Debug, PartialEq)]
pub struct Vol<T> {
    pub data: Vec<T>,
    pub shape: [usize; 4],
}

impl<T: Copy + Default> Vol<T> {
    pub fn zeros(shape: [usize; 4]) -> Self {
        Vol { data: vec![T::default(); shape.iter().product()], shape }
    }

    pub fn from_vec(data: Vec<T>, shape: [usize; 4]) -> Self {
        assert_eq!(data.len(), shape.iter().product::<usize>(), "data does not match shape");
        Vol { data, shape }
    }

    pub fn nx(&self) -> usize {
        self.shape[0]
    }
    pub fn ny(&self) -> usize {
        self.shape[1]
    }
    pub fn nz(&self) -> usize {
        self.shape[2]
    }
    pub fn nt(&self) -> usize {
        self.shape[3]
    }

    #[inline]
    pub fn idx(&self, x: usize, y: usize, z: usize, t: usize) -> usize {
        let [nx, ny, nz, _] = self.shape;
        x + nx * (y + ny * (z + nz * t))
    }

    #[inline]
    pub fn get(&self, x: usize, y: usize, z: usize, t: usize) -> T {
        self.data[self.idx(x, y, z, t)]
    }

    /// One slice of one frame, element (x, y) at x + nx * y.
    pub fn slice(&self, z: usize, t: usize) -> &[T] {
        let n = self.nx() * self.ny();
        let off = (z + self.nz() * t) * n;
        &self.data[off..off + n]
    }

    pub fn slice_mut(&mut self, z: usize, t: usize) -> &mut [T] {
        let n = self.nx() * self.ny();
        let off = (z + self.nz() * t) * n;
        &mut self.data[off..off + n]
    }

    /// All slices of one frame, element (x, y, z) at x + nx * (y + ny * z).
    pub fn frame(&self, t: usize) -> &[T] {
        let n = self.nx() * self.ny() * self.nz();
        &self.data[t * n..(t + 1) * n]
    }

    pub fn frame_mut(&mut self, t: usize) -> &mut [T] {
        let n = self.nx() * self.ny() * self.nz();
        &mut self.data[t * n..(t + 1) * n]
    }

    /// The sub-volume x0..x1, y0..y1 (all slices and frames).
    pub fn crop_xy(&self, x0: usize, x1: usize, y0: usize, y1: usize) -> Vol<T> {
        let [_, _, nz, nt] = self.shape;
        let (w, h) = (x1 - x0, y1 - y0);
        let mut data = Vec::with_capacity(w * h * nz * nt);
        for t in 0..nt {
            for z in 0..nz {
                let sl = self.slice(z, t);
                for y in y0..y1 {
                    let row = y * self.nx();
                    data.extend_from_slice(&sl[row + x0..row + x1]);
                }
            }
        }
        Vol { data, shape: [w, h, nz, nt] }
    }

    /// Frame t as a 4D volume with one frame.
    pub fn frame_vol(&self, t: usize) -> Vol<T> {
        let [nx, ny, nz, _] = self.shape;
        Vol { data: self.frame(t).to_vec(), shape: [nx, ny, nz, 1] }
    }

    pub fn map<U: Copy + Default>(&self, f: impl Fn(T) -> U) -> Vol<U> {
        Vol { data: self.data.iter().map(|&v| f(v)).collect(), shape: self.shape }
    }
}

/// A 2D view of one slice: element (x, y) at x + nx * y.
#[derive(Clone, Copy)]
pub struct Slice2<'a, T> {
    pub data: &'a [T],
    pub nx: usize,
    pub ny: usize,
}

impl<'a, T: Copy> Slice2<'a, T> {
    pub fn new(data: &'a [T], nx: usize, ny: usize) -> Self {
        debug_assert_eq!(data.len(), nx * ny);
        Slice2 { data, nx, ny }
    }
    #[inline]
    pub fn get(&self, x: usize, y: usize) -> T {
        self.data[x + self.nx * y]
    }
}

impl Vol<u8> {
    pub fn slice2(&self, z: usize, t: usize) -> Slice2<'_, u8> {
        Slice2::new(self.slice(z, t), self.nx(), self.ny())
    }
}
