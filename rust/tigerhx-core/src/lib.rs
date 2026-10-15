//! UI-independent TigerHx processing, a port of the Python `tigerhx.core`.
//!
//! ```text
//! nifti      NIfTI-1/2 reading, label maps, display volumes
//! inference  ONNX segmentation of cine images
//! aha        AHA sectors, slice order and apex detection
//! metrics    ED/ES frames and segmentation checks
//! legacy     names, zoom box and the original GUI's .mat result
//! mat        MATLAB 5 file writer and reader
//! caselist   the original GUI's case-list CSV
//! models     model files and download
//! project    workspace with a list of cases
//! pipeline   per-case processing used by the GUI
//! ```

pub mod aha;
pub mod caselist;
pub mod error;
pub mod fsutil;
pub mod inference;
pub mod legacy;
pub mod mat;
pub mod metrics;
pub mod models;
pub mod nifti;
pub mod pipeline;
pub mod project;
pub mod volume;

pub use error::{Error, Result};
pub use volume::Vol;
