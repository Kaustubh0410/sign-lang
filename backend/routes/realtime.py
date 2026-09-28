from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from utils.pose_estimation import PoseEstimator
from utils.translator import SignTranslator
import json
import time
import traceback

router = APIRouter()

pose_estimator = PoseEstimator(sequence_length=30)  # 1 second at 30fps
translator = SignTranslator()

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    print("WebSocket connected")
    
    try:
        while True:
            # Receive frame data
            data = await websocket.receive_bytes()
            print(f"Received frame data: {len(data)} bytes")
            
            try:
                # Get temporal sequence for prediction
                keypoint_sequence = pose_estimator.get_sequence_for_prediction(data)
                print(f"Keypoint sequence: {keypoint_sequence.shape if keypoint_sequence is not None else 'None'}")
                
                if keypoint_sequence is not None:
                    # Perform complete translation
                    translation_result = translator.translate_frame_sequence(keypoint_sequence)
                    print(f"Translation result: {translation_result}")
                    
                    # Send result as JSON
                    response = {
                        "timestamp": time.time(),
                        "gloss_sequence": translation_result["gloss_sequence"],
                        "english_translation": translation_result["english_translation"],
                        "confidence": translation_result["confidence"]
                    }
                    
                    await websocket.send_text(json.dumps(response))
                    print("Sent translation response")
                else:
                    # Send status update while building sequence
                    status_response = {
                        "timestamp": time.time(),
                        "status": "building_sequence",
                        "frames_collected": len(pose_estimator.keypoint_sequence),
                        "frames_needed": pose_estimator.sequence_length
                    }
                    await websocket.send_text(json.dumps(status_response))
                    print(f"Sent status: {status_response}")
                    
            except Exception as e:
                print(f"Error processing frame: {e}")
                print(traceback.format_exc())
                # Send error response
                error_response = {
                    "timestamp": time.time(),
                    "error": str(e),
                    "status": "error"
                }
                await websocket.send_text(json.dumps(error_response))
                
    except WebSocketDisconnect:
        print("WebSocket disconnected")
        pose_estimator.reset_sequence()
    except Exception as e:
        print(f"WebSocket error: {e}")
        print(traceback.format_exc())
        pose_estimator.reset_sequence()

@router.post("/reset")
async def reset_sequence():
    """Reset the temporal sequence"""
    pose_estimator.reset_sequence()
    return {"status": "sequence_reset"}

