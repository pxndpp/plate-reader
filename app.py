import gradio as gr
import cv2

from pipeline import read_plates

#แปลงสีจาก rgb เป็น bgr ก่อนเพราะ ai เรารับแค่ bgr แต่ ui เรารับเป็น rgb เดี๋ยวสีมันเพี้ยน
def run(img_rgb):
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    plates = read_plates(img_bgr)
    return [[i, p["registration"], p["province"]] for i, p in enumerate(plates, start=1)]

with gr.Blocks() as demo:
    gr.Markdown("THAI PLATE READER")
    photo = gr.Image(label="Car pic", type="numpy")
    table = gr.DataFrame(headers=["no","Plates", "Province"], interactive=False)
    gr.Button("Read this plate").click(run, inputs=photo, outputs=table)
    
demo.launch()