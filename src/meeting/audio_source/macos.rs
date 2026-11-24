//! macOS's meeting audio sources: the Core Audio processes playing output
//! and the listed output devices.

use super::SourceEntry;

/// The applications playing output now, read from Core Audio's process
/// metadata (never audio). A helper process is named as its app, once.
pub(super) fn playing_apps() -> Vec<SourceEntry> {
    crate::microphone_activity::playing_output_applications()
        .into_iter()
        .map(|(id, name)| SourceEntry { id, name })
        .collect()
}

/// The output devices, the default first. The id is Core Audio's device UID.
pub(super) fn output_devices() -> Vec<SourceEntry> {
    super::listed_outputs()
}
