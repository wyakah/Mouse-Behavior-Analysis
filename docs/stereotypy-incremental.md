# Incremental scoring

After timestamp indexing and model initialization, the automatic worker infers
eight-second sections and immediately renders their scored frames. Three future
priority samples preserve the seven-sample temporal context. Two-second video
segments publish while later sections remain unprocessed. Source gaps remain
unknown, and the 1,200-second limit applies throughout.

The main model weights, thresholds, DINOv2 features, geometry and weak-head
features are unchanged. Motion now uses the same PyAV-decoded frames as the
visual features and annotated video. The former standalone motion pass used
OpenCV decoding. Small decoder differences can shift scores near a threshold.

On mouse 756, the first section was emitted after model input through 9.1 seconds,
and the second after input through 17.1 seconds. The full run retained all 36,000
source frames and produced 600 live segments. Compared with the previous run,
grooming and digging each increased by 0.5 seconds; other activity decreased by
1.0 second. The other four behavior totals and unknown time were unchanged.
This is an implementation comparison, not an accuracy evaluation.

Timestamp caches require identical video content, the current timing recipe,
and the original index digest. They do not reuse predictions or bypass inference.
Single-video previews expand automatically; multiple-video previews remain
collapsed until opened. Scientific method details remain in exports and these
documents rather than repeated throughout the working UI.
