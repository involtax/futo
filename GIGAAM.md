# Built-in Russian live dictation

Russian voice input uses **GigaAM v2 CTC INT8** through **sherpa-onnx 1.12.40**.
The model, character vocabulary, native runtime and licenses are included in the APK.
No speech-model download or speech service is needed after installation.
Other languages retain the existing Whisper backend.

Select the Russian keyboard layout and use its microphone button. The built-in
voice input backend must be selected in voice input settings. **Live Russian
dictation** is on by default. Turn it off to compare whole-recording recognition.
In live mode pauses finish phrases without stopping the microphone; tap the voice
input panel to stop. The current phrase may be corrected while more audio arrives.
GigaAM v2 CTC does not automatically add punctuation or normalize numbers.

## Implementation

The microphone continues reading 16 kHz mono PCM while a single recognizer processes
snapshots. Requests are conflated, not queued per audio frame. Previews update after
approximately one additional second of audio, subject to inference speed. A pause
of approximately 700 ms finalizes a phrase. Continuous speech uses at most 12 seconds
per inference window and rolls at a decoded word boundary, retaining its tail and
up to two seconds of previously decoded audio as context. That overlap is removed
from the next transcript without removing intentional repeated words.
Pending raw audio is retained if inference is slower than recording. This is
simulated streaming with an offline CTC model, not a cache-aware streaming model.

`setComposingText` replaces provisional dictation in the editor; the final
transaction is committed on stop. Non-text editors may only show the final result.
The model's user glossary/punctuation settings are not supported by this CTC backend.

## Sources and integrity

- Model: https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/sherpa-onnx-nemo-ctc-giga-am-v2-russian-2025-04-19.tar.bz2
- Original model: https://github.com/salute-developers/GigaAM (MIT).
- Model archive SHA-256: `777be8717d8aaf04861823671290f7687f7579fd9ac63a2124955573f920caf5`.
- Packaged ONNX SHA-256: `d0ce4aef25f58d495781ee8f05320d9e51b821f47804e07aa6549b53a72f67e8`.
- Android runtime: https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.12.40/sherpa-onnx-1.12.40.aar (Apache-2.0).
- AAR SHA-256: `a7699d6a2e638a38791b2459cb0099c7577e6e5fca6b8ae916742c872a618b82`.

The model and AAR use Git LFS. After cloning, run `git lfs pull` in the root,
`git submodule update --init --recursive`, and Git LFS pull in the swipe submodule.
Use the environment setup instructions for the pinned JDK and Android SDK.

## Build and tests

With the Android SDK and full JDK initialized:

```sh
VERSION_CODE=2 VERSION_NAME=0.0.1-gigaam ./gradlew --max-workers=4 assembleUnstableDebug
./gradlew :voiceinput-shared:testDebugUnitTest
```

Buffer tests run without native libraries. Native inference tests require the
model archive above (including `test_wavs/example.wav`) and the matching Linux
x64 JNI package:
https://github.com/k2-fsa/sherpa-onnx/releases/download/v1.12.40/sherpa-onnx-v1.12.40-linux-x64-jni.tar.bz2
SHA-256: `6ace40005534a65110e990ce3a0afa5ed16acb50a04f9328013cc90ec8e52516`.

```sh
GIGAAM_TEST_MODEL_DIR=/path/to/extracted/model \
SHERPA_JNI_DIR=/path/to/extracted/jni/lib \
./gradlew :voiceinput-shared:testDebugUnitTest
```

Native tests exercise the application's Kotlin decoder, whole-recording Russian
recognition, previews before stop, and repeated speech across rolling windows.
They do not replace Android microphone/editor tests. On a phone, check permissions,
Russian dictation in at least two apps, a pause followed by resumed speech, a long
utterance, stopping during recognition, and closing/reopening the keyboard. Check
airplane mode to confirm that the packaged model is sufficient.
