from feat import Detector

detector = Detector(
    face_model="retinaface",
    landmark_model="mobilefacenet",
    au_model='xgb',
    emotion_model="resmasknet",
    facepose_model="img2pose",
)

#detector

single_face_img_path = ("frames_output/1_1.jpg")

single_face_prediction = detector.detect_image(single_face_img_path)

# Show results
print(single_face_prediction)