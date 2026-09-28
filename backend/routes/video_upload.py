from fastapi import APIRouter, UploadFile, File, HTTPException
from utils.frame_extraction import extract_frames
from utils.pose_estimation import PoseEstimator
from utils.translator import SignTranslator

router = APIRouter()

pose_estimator = PoseEstimator()
translator = SignTranslator()

@router.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    if not file.filename.lower().endswith((".mp4", ".avi", ".mov")):
        raise HTTPException(status_code=400, detail="Invalid file type")

    frames = await extract_frames(file)
    gloss_sequence = []
    for frame in frames:
        keypoints = pose_estimator.extract_keypoints(frame)
        gloss_sequence.extend(translator.predict_gloss(keypoints))

    sentence = translator.gloss_to_sentence(gloss_sequence)
    return {"translation": sentence}
