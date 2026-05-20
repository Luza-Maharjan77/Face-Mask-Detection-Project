import cv2
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

# ----------------------------
# MODEL ARCHITECTURE
# ----------------------------
class ImprovedCNN(nn.Module):
    def __init__(self, num_classes=3):
        super(ImprovedCNN, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(128 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


# ----------------------------
# SETUP
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

classes = ["mask_weared_incorrect", "with_mask", "without_mask"]

model = ImprovedCNN(len(classes)).to(device)
model.load_state_dict(torch.load("improved_model.pth", map_location=device))
model.eval()

# ----------------------------
# IMAGE TRANSFORM
# ----------------------------
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# ----------------------------
# FACE DETECTOR
# ----------------------------
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

profile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_profileface.xml"
)
# ----------------------------
# CAMERA
# ----------------------------
cap = cv2.VideoCapture(0)

print("Webcam started... Press Q to exit")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 3)

    profiles = profile_cascade.detectMultiScale(gray, 1.1, 3)

    faces = list(faces) + list(profiles)

    if len(faces) == 0:
        cv2.putText(frame, "No face detected", (30, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.imshow("Mask Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        continue

    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]

        # Convert for model
        face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
        face = Image.fromarray(face)

        input_tensor = transform(face).unsqueeze(0).to(device)

        # Prediction
        with torch.no_grad():
            outputs = model(input_tensor)
            _, pred = torch.max(outputs, 1)
            label = classes[pred.item()]

        # Draw box + label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame, label, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0), 2)

    cv2.imshow("Mask Detection", frame)
    
    key = cv2.waitKey(1) & 0xFF

    # Press Q OR click X
    if key == ord('q'):
        break

    # If window is closed manually
    if cv2.getWindowProperty("Mask Detection", cv2.WND_PROP_VISIBLE) < 1:
        break

cap.release()
cv2.destroyAllWindows()