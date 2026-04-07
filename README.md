# Smart Retail Edge AI 🛒🤖

This project is me trying to learn and build a **real-world Edge AI system for retail scenarios**, step by step, a bit more practical instead of just learning concepts in isolation. Have been using AI tools to assist with learning pathway and understanding the workflow and implementation better and clearly, rather than getting stuck/lost too deep in less important areas.

The idea is simple:

Take camera input → process it locally using AI → understand what’s happening → and eventually turn that into useful actions (like alerts, insights, or automation).

Right now, I’m still building it layer by layer, but the goal is to simulate how a real smart retail system would work.

---

## 🧠 What I’ve Built So Far

Started from scratch with:

* Python virtual environment setup (clean project setup)
* Integrated YOLO (Ultralytics) for real-time object detection
* Built a live webcam detection system using OpenCV
* Displayed bounding boxes with labels and confidence scores
* Understood how inference works locally (Edge AI concept)

Then moved into the next phase:

* Implemented face detection (initially using Haar Cascade)

* Observed real-world issues like:

  * false detections
  * instability in the model's output/detection
  * sensitivity to angles and lighting

* Upgraded to a better face detection approach (YuNet / OpenCV DNN)

  * more stable detections
  * better accuracy
  * added facial keypoints

---

## 🚀 Current Direction

Now the project is moving from:

**“detecting objects” → “understanding people and behavior”**

The next actionable steps that I’m actively working on:

* Face tracking (following a face across frames)
* Temporary individual IDs (Face 1, Face 2, etc.)
* Basic attribute estimation (gender / expression — carefully, not overclaiming yet)
* Zone-based logic (like shelf interaction areas)
* Converting detections into meaningful events

---

## 🧰 Tech Stack

* Python 🐍
* OpenCV (video processing + visualization)
* YOLO (Ultralytics) for object detection
* OpenCV DNN (YuNet) for face detection
* VS Code
* GitHub (version control + documentation)
* GitHub Copilot (coding assistance)

---

## ⚙️ How to Run

### 1. Clone the repo

```bash
git clone https://github.com/your-username/smart-retail-edge-ai.git
cd smart-retail-edge-ai
```

### 2. Create virtual environment

```bash
python -m venv venv
venv\Scripts\activate   # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run scripts

Object detection:

```bash
python quick_YOLO_test.py
```

Face detection:

```bash
python face_detection.py
```

---

## 📌 What I learned so far in my approach

* Edge AI is not just about running models — it’s about **latency, reliability, and real-time behavior**
* Detection alone is not useful until you add a viable **logic on top**
* Model choice matters a lot (Haar vs DNN difference was very clear)
* Debugging real-time systems is very different from static code
* Building step-by-step helps more than jumping into complex systems

---

## ⚠️ Current Limitations

* Face detection still depends on lighting and camera quality
* No persistent identity tracking yet (only detection per frame)
* Attribute estimation not implemented yet
* No logging or dashboard yet

---

## 🔮 Future Improvements

* Face tracking and consistent IDs
* Behavior/event detection (e.g. customer near shelf)
* Logging system (CSV / database)
* Simple dashboard or webpage for visualization
* Better models if needed (MediaPipe / custom models)

---

## 📸 (To Add)

* Screenshots of detection output
* Short demo video
* System flow diagram

---

## 💭 Why I Built This

I didn’t want to just “learn AI tools” and I didn't want to just chase the output — I wanted to **build something that feels like actual work and shows consistent lasting progress and understanding of everything I am learning and integrating during**.

This project is my way of:

* understanding Edge AI practically
* connecting AI + IoT + real-world use cases
* preparing for real industry scenarios

Still learning, still building 🚀
