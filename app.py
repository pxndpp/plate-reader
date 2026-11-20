import gradio as gr
import cv2

from pipeline import read_plates

def run(img_rgb):
    
    if img_rgb is None:
        raise gr.Error("UPLOAD PIC FIRST KUB!")
    #แปลงสีจาก rgb เป็น bgr ก่อนเพราะ ai เรารับแค่ bgr แต่ ui เรารับเป็น rgb เดี๋ยวสีมันเพี้ยน
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    plates = read_plates(img_bgr)
    annotated = cv2.cvtColor(draw_boxes(img_bgr, plates),cv2.COLOR_BGR2RGB)
    
    crop = [(cv2.cvtColor(p["crop"], cv2.COLOR_BGR2RGB), f"Plate no: {i}") for i, p in enumerate (plates, start=1)]
    
    table = [
        [i, p["registration"], p["province"],
         f"{p['detection_confidence']:.0%}", f"{p['reading_confidence']:.0%}"]
        for i, p in enumerate(plates, start=1)
    ]
    message = f"Found {len(plates)} plates" if plates else "Plate not found in this pic!"
    return annotated, crop, table, message


def draw_boxes(image_bgr, plates):
    #วาดกรอบสีเขียวพร้อมเลขลำดับ
    out = image_bgr.copy()
    thickness = max(2, out.shape[1] // 300)  # ภาพใหญ่ให้เส้นหนาขึ้น
    for number, plate in enumerate(plates, start=1):
        x1, y1, x2, y2 = plate["bbox"]
        cv2.rectangle(out, (x1, y1), (x2, y2), (0, 200, 0), thickness)
        cv2.putText(out, "Plate:"+str(number), (x1, max(y1 - 10, 30)),
                    cv2.FONT_HERSHEY_SIMPLEX, thickness / 2, (0, 200, 0), thickness)
    return out


with gr.Blocks() as demo:
    gr.Markdown("THAI PLATE READER")
    with gr.Row():
        photo = gr.Image(label="Car pic", type="numpy")
        result_photo = gr.Image(label="Plates that's found!")
    
    crops = gr.Gallery(label="Found plates", columns=3, height=260)
    table = gr.DataFrame(headers=["Plate no.","Plates", "Province", "Detection Confidence", "Read Confidence"], interactive=False)
    message = gr.Markdown("")
    gr.Button("Read this plate").click(run, inputs=photo, outputs=[result_photo, crops, table, message])

if __name__ == "__main__":
    demo.launch()