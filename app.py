import streamlit as st
import cv2
import supervision as sv
from inference.models.yolo_world.yolo_world import YOLOWorld
import numpy as np
import tempfile
from dotenv import load_dotenv
from streamlit_drawable_canvas import st_canvas
from PIL import Image

# Load environment variables
load_dotenv()

st.set_page_config(page_title="Real-time Plate Counter", layout="wide")
st.title("YOLO-World Plate Counter")

# Sidebar Configuration
st.sidebar.header("Configuration")

# Model Configuration
model_id = st.sidebar.selectbox("Model Version", ["yolo_world/l", "yolo_world/m", "yolo_world/s", "yolo_world/x"], index=0)
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.01, 0.01)
iou_threshold = st.sidebar.slider("IoU Threshold", 0.0, 1.0, 0.5, 0.01)
frame_skip = st.sidebar.slider("Frame Skip Rate (Speed)", 1, 5, 2, help="Process 1 out of every N frames. Higher = Faster but choppier.")
classes_input = st.sidebar.text_input("Classes (comma separated)", "plate, license plate")
class_list = [c.strip() for c in classes_input.split(",")]

# Tabs for Setup and Run
tab1, tab2 = st.tabs(["Run Inference", "Setup Line Zone"])

# Initialize Session State
if "line_coords" not in st.session_state:
    st.session_state.line_coords = {
        "start": (0, 360), 
        "end": (1280, 360),
        "norm_start": (0.0, 0.5),
        "norm_end": (1.0, 0.5)
    }

if "setup_frame" not in st.session_state:
    st.session_state.setup_frame = None

# Initialize Model
@st.cache_resource
def load_model(model_id):
    print(f"Loading YOLO-World model: {model_id}...")
    return YOLOWorld(model_id=model_id)

model = load_model(model_id)

# Initialize Tracker and Annotators
tracker = sv.ByteTrack()
box_annotator = sv.BoxAnnotator(thickness=4)
label_annotator = sv.LabelAnnotator(text_scale=1.5, text_thickness=3)
trace_annotator = sv.TraceAnnotator(thickness=4)
# Use Green color for visibility
line_zone_annotator = sv.LineZoneAnnotator(
    thickness=4, 
    text_thickness=4, 
    text_scale=2,
    color=sv.Color(r=0, g=255, b=0)
)

# Source Selection
source_type = st.sidebar.radio("Select Video Source", ["Sample Image", "Upload Video", "Webcam"])

def capture_frame():
    if source_type == "Sample Image":
        return cv2.imread("IMG_8144.png")
    elif source_type == "Upload Video":
        uploaded_file = st.sidebar.file_uploader("Choose a video...", type=["mp4", "mov", "avi"], key="video_uploader")
        if uploaded_file is not None:
            tfile = tempfile.NamedTemporaryFile(delete=False)
            tfile.write(uploaded_file.read())
            cap = cv2.VideoCapture(tfile.name)
            ret, frame = cap.read()
            cap.release()
            return frame if ret else None
    elif source_type == "Webcam":
        cap = cv2.VideoCapture(0)
        # Warm up
        for _ in range(5): cap.read()
        ret, frame = cap.read()
        cap.release()
        return frame if ret else None
    return None

# --- SETUP TAB ---
with tab2:
    st.header("Draw Your Line Zone")
    st.info("1. Click 'Capture Frame'. 2. Draw a line. 3. Click 'Save Zone'.")

    if st.button("Capture Frame for Setup"):
        frame = capture_frame()
        if frame is not None:
             # Convert BGR to RGB
            st.session_state.setup_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # Create PIL Image once and store it to prevent re-creation loops
            st.session_state.setup_pil = Image.fromarray(st.session_state.setup_frame)
            # Clear previous canvas state if needed by changing key? No, just keep simple.
        else:
            st.error("Could not capture frame. Check source.")

    if st.session_state.setup_frame is not None:
        # Resize frame for canvas (Fixes coordinate scaling mismatch on high-res screens)
        # We enforce a max width (e.g. 800px) so the canvas 1:1 matches the display
        preview_width = 800
        h, w = st.session_state.setup_frame.shape[:2]
        aspect_ratio = w / h
        preview_height = int(preview_width / aspect_ratio)
        
        # Resize just for display/canvas interaction
        preview_frame = cv2.resize(st.session_state.setup_frame, (preview_width, preview_height))
        pil_img = Image.fromarray(preview_frame)

        # Create canvas with explicit display dimensions
        # Key: width/height here define the coordinate system of the canvas
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",
            stroke_width=5,
            stroke_color="#FF0000",
            background_image=pil_img,
            update_streamlit=True,
            height=preview_height,
            width=preview_width,
            drawing_mode="line",
            key="canvas",
        )

        # Button to explicitly save the line
        if st.button("Save Zone", type="primary"):
            if canvas_result.json_data is not None:
                objects = canvas_result.json_data["objects"]
                if objects:
                    last_obj = objects[-1]
                    if last_obj["type"] == "line":
                        # standard fabric.js behavior with center origin:
                        center_x = last_obj.get("left", 0)
                        center_y = last_obj.get("top", 0)
                        
                        rel_x1 = last_obj.get("x1", 0)
                        rel_y1 = last_obj.get("y1", 0)
                        rel_x2 = last_obj.get("x2", 0)
                        rel_y2 = last_obj.get("y2", 0)
                        
                        start_x = center_x + rel_x1
                        start_y = center_y + rel_y1
                        end_x = center_x + rel_x2
                        end_y = center_y + rel_y2
                        
                        # Calculate normalized coordinates (0.0 to 1.0) Using PREVIEW dimensions
                        norm_start_x = start_x / preview_width
                        norm_start_y = start_y / preview_height
                        norm_end_x = end_x / preview_width
                        norm_end_y = end_y / preview_height
                        
                        # We don't save absolute pixels here because they are "preview pixels"
                        # But session_state.line_coords["start"] is used for non-normalized fallback?
                        # Let's map them back to ORIGINAL resolution just in case legacy logic needs them
                        # BUT reliance on normalized is preferred.
                        
                        # Map preview-pixel coords back to original-resolution pixels for "start"/"end" fields
                        orig_start_x = int(norm_start_x * w)
                        orig_start_y = int(norm_start_y * h)
                        orig_end_x = int(norm_end_x * w)
                        orig_end_y = int(norm_end_y * h)
                        
                        new_coords = {
                            "start": (orig_start_x, orig_start_y),
                            "end": (orig_end_x, orig_end_y),
                            "norm_start": (norm_start_x, norm_start_y),
                            "norm_end": (norm_end_x, norm_end_y)
                        }
                        st.session_state.line_coords = new_coords
                        st.success(f"Saved: {new_coords['start']} -> {new_coords['end']}")
                else:
                    st.warning("No line drawn on canvas.")
    else:
        st.warning("No frame captured yet. Click the button above.")

# --- RUN TAB ---
with tab1:
    st.header("Real-time Inference")
    
    # Initialize session state line zone if missing
    if "line_zone" not in st.session_state:
        # Default fallback if no coords set yet
        st.session_state.line_zone = sv.LineZone(
            start=sv.Point(0, 360),
            end=sv.Point(1280, 360)
        )

    # We will update the LineZone inside the loop or just before if frame shape is known.
    # Since we need the current frame shape to denormalize, we can't do it blindly here.
    # We'll delegate LineZone update to _process_logic where we have the frame.
    
    if "run" not in st.session_state:
        st.session_state.run = False

    def start_stop():
        st.session_state.run = not st.session_state.run

    st.button("Start/Stop Inference", on_click=start_stop)
    
    st.write(f"Stored Config: {st.session_state.line_coords.get('start', 'Default')} (Normalized Available: {'norm_start' in st.session_state.line_coords})")
    
    # Placeholder for live active coordinates
    coords_placeholder = st.empty()
    placeholder = st.empty()
    metric_placeholder = st.empty()
    
    
    
    def process_frame(frame, line_zone_instance):
        # Calculate scale factor based on resolution (Reference width: 1280)
        height, width = frame.shape[:2]
        scale_factor = width / 1280.0
        
        # Dynamic Thickness and Text Scale
        dynamic_thickness = max(1, int(4 * scale_factor))
        dynamic_text_scale = max(0.5, 1.5 * scale_factor)
        dynamic_text_thickness = max(1, int(2 * scale_factor))
        
        # Re-init stateless annotators with dynamic sizing
        box_annotator = sv.BoxAnnotator(thickness=dynamic_thickness)
        label_annotator = sv.LabelAnnotator(
            text_scale=dynamic_text_scale, 
            text_thickness=dynamic_text_thickness
        )
        line_zone_annotator = sv.LineZoneAnnotator(
            thickness=dynamic_thickness, 
            text_thickness=dynamic_thickness, 
            text_scale=dynamic_text_scale,
            color=sv.Color(r=0, g=255, b=0)
        )
        
        # Trace annotator has state, treat carefully or check if we can update properties
        # For now, we'll keep it simple or try to update if attribute exists, else leave default
        # sv.TraceAnnotator usually has .thickness attribute
        if hasattr(trace_annotator, "thickness"):
            trace_annotator.thickness = dynamic_thickness

        # Check and update LineZone based on current frame resolution for POSITIONAL scaling
        # Check if we have normalized coords
        if "norm_start" in st.session_state.line_coords:
            ns = st.session_state.line_coords["norm_start"]
            ne = st.session_state.line_coords["norm_end"]
            
            # Calculate expected absolute coords
            abs_start_x = int(ns[0] * width)
            abs_start_y = int(ns[1] * height)
            abs_end_x = int(ne[0] * width)
            abs_end_y = int(ne[1] * height)
            
            # Strategy: Store 'last_shape' and 'last_line_coords' in session state. 
            # If changed, re-init.
            current_coords = st.session_state.line_coords
            last_coords = st.session_state.get("last_line_coords", {})
            
            shape_changed = "last_shape" not in st.session_state or st.session_state.last_shape != (width, height)
            coords_changed = current_coords != last_coords
            
            if shape_changed or coords_changed:
                print(f"Update Triggered: Shape={shape_changed}, Coords={coords_changed}")
                print(f"Rescaling LineZone to {width}x{height}")
                
                st.session_state.line_zone = sv.LineZone(
                    start=sv.Point(abs_start_x, abs_start_y),
                    end=sv.Point(abs_end_x, abs_end_y)
                )
                st.session_state.last_shape = (width, height)
                st.session_state.last_line_coords = current_coords.copy()
        
        # Display effective coordinates
        lz = st.session_state.line_zone
        coords_placeholder.markdown(f"**Effective Line Zone (Pixels):** Start `{lz.vector.start}` End `{lz.vector.end}` | **Resolution:** `{width}x{height}`")
                
        # Use the (possibly updated) line_zone from session state
        return_frame = _process_logic(
            frame, lz, 
            box_annotator, label_annotator, trace_annotator, line_zone_annotator
        )
        return return_frame

    def _process_logic(frame, line_zone_instance, box_ann, label_ann, trace_ann, lz_ann):
        results = model.infer(frame, text=class_list, confidence=confidence_threshold)
        if isinstance(results, list):
            result = results[0]
        else:
            result = results

        detections = sv.Detections.from_inference(result)
        detections = detections.with_nms(threshold=iou_threshold)
        detections = tracker.update_with_detections(detections)
        
        line_zone_instance.trigger(detections=detections)
        
        labels = []
        for tracker_id, class_id, confidence in zip(detections.tracker_id, detections.class_id, detections.confidence):
            labels.append(f"#{tracker_id} {class_list[class_id]} {confidence:0.2f}")
        
        annotated_frame = box_ann.annotate(scene=frame.copy(), detections=detections)
        annotated_frame = label_ann.annotate(scene=annotated_frame, detections=detections, labels=labels)
        annotated_frame = trace_ann.annotate(scene=annotated_frame, detections=detections)
        annotated_frame = lz_ann.annotate(annotated_frame, line_counter=line_zone_instance)
        
        return annotated_frame

    if st.session_state.run:
        lz = st.session_state.line_zone
        
        if source_type == "Sample Image":
            image_path = "IMG_8144.png"
            frame = cv2.imread(image_path)
            if frame is not None:
                # Removed crashing debug line for production
                annotated_frame = process_frame(frame, lz)
                # FIX: use_column_width instead of use_container_width
                placeholder.image(annotated_frame, channels="BGR", use_column_width=True)
                metric_placeholder.metric("Count Out", lz.out_count)
            else:
                st.error("Image not found.")
                
        elif source_type == "Upload Video":
            # Just grab the file again from sidebar widget state if possible, or we need to rely on the uploader above
             # NOTE: Re-declaring here to ensure access. Streamlit will keep state.
            uploaded_file = st.sidebar.file_uploader("Choose a video...", type=["mp4", "mov", "avi"], key="video_uploader")
            if uploaded_file is not None:
                # We need to write to temp file again or manage it better? 
                # Ideally check if we already have a temp file path in session state, but for now simple re-write is safer for robust path access.
                tfile = tempfile.NamedTemporaryFile(delete=False)
                tfile.write(uploaded_file.read())
                
                cap = cv2.VideoCapture(tfile.name)
                frame_count = 0
                while st.session_state.run and cap.isOpened():
                    ret, frame = cap.read()
                    if not ret: break
                    
                    frame_count += 1
                    if frame_count % frame_skip != 0:
                        continue
                        
                    annotated_frame = process_frame(frame, lz)
                    # FIX: use_column_width
                    placeholder.image(annotated_frame, channels="BGR", use_column_width=True)
                    metric_placeholder.metric("Count Out", lz.out_count)
                cap.release()
            else:
                 st.info("Upload a video in the sidebar.")
        
        elif source_type == "Webcam":
            cap = cv2.VideoCapture(0)
            frame_count = 0
            while st.session_state.run and cap.isOpened():
                ret, frame = cap.read()
                if not ret: break
                
                frame_count += 1
                if frame_count % frame_skip != 0:
                    continue

                annotated_frame = process_frame(frame, lz)
                # FIX: use_column_width
                placeholder.image(annotated_frame, channels="BGR", use_column_width=True)
                metric_placeholder.metric("Count Out", lz.out_count)
            cap.release()

# Correction for Video Upload logic in the main flow
if source_type == "Upload Video" and st.session_state.run:
    pass
