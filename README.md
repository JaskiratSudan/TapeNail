# TapeNail: Secure Your Phone Access Through Your Nail

**TapeNail** is a hardware-software authentication system that uses passive 3D optical patterns embedded in nail art to unlock smartphones. The system leverages polarized light, real-time camera capture, and a lightweight deep learning model to detect and verify unique, user-defined optical signatures without relying on permanent biometrics.

---

## Repository Structure

| Folder                     | Description                     |
|---------------------------|---------------------------------|
| `ANDROID_APP/`            | Mobile app code |
| `SCRIPTS/`       | Prototype code  |
| `DOCUMENTS/` | Docs, drafts, figures, notes     |
| `TAPENAIL_YOLO/`          | YOLO-based pattern detection    |

---

## Features

- Revocable authentication with passive optical tokens
- Personalized pattern design (3D + layered aesthetics)
- Secure against spoofing, cloning, and replay attacks
- On-device YOLOv11-n model (lightweight, fast)
- No additional hardware — works with phone camera

---

## Tooling

- **Android**: Java / Kotlin (Camera2 API, TFLite)
- **Model Training**: YOLOv11-n using Ultralytics
- **Dataset Management**: Roboflow + manual augmentation
- **Pattern Design**: Transparent tape, polarized films, glitter base

---

## Getting Started

Clone the repository and navigate into the Android app:

```bash
git clone https://github.com/your-org/TapeNail.git
cd TapeNail/ANDROID_APP



