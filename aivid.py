import streamlit as st
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
import torch
from diffusers import DiffusionPipeline, DPMSolverMultistepScheduler
import tempfile
import os
from PIL import Image
import base64

def create_download_link(video_path):
    """إنشاء رابط لتحميل الفيديو"""
    with open(video_path, 'rb') as f:
        bytes = f.read()
        b64 = base64.b64encode(bytes).decode()
        href = f'<a href="data:file/mp4;base64,{b64}" download="generated_video.mp4">اضغط هنا لتحميل الفيديو</a>'
        return href

def generate_video(prompt, duration=5):
    """توليد فيديو باستخدام نموذج الذكاء الاصطناعي"""
    # تهيئة النموذج باستخدام إعدادات خفيفة
    pipe = DiffusionPipeline.from_pretrained(
        "damo-vilab/text-to-video-ms-1.7b",
        torch_dtype=torch.float16,
        variant="fp16"
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    else:
        pipe = pipe.to("cpu")
    
    # توليد الفيديو
    video_frames = pipe(
        prompt,
        num_inference_steps=25,
        num_frames=duration * 8
    ).frames

    # حفظ الفيديو مؤقتاً
    temp_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_dir, "generated_video.mp4")
    
    # تحويل الإطارات إلى فيديو
    video_clip = VideoFileClip(video_frames)
    video_clip.write_videofile(output_path, fps=8, codec='libx264')
    
    return output_path

def main():
    st.title("منشئ الفيديوهات بالذكاء الاصطناعي")
    
    # إدخال النص الوصفي
    prompt = st.text_input("أدخل وصفاً للفيديو الذي تريد إنشاءه", "مشهد لغروب الشمس على الشاطئ")
    
    # إعدادات إضافية
    duration = st.slider("مدة الفيديو (بالثواني)", 3, 10, 5)
    
    if st.button("إنشاء الفيديو"):
        with st.spinner('جاري إنشاء الفيديو...'):
            try:
                video_path = generate_video(prompt, duration)
                
                # عرض الفيديو
                st.video(video_path)
                
                # إضافة زر التحميل
                st.markdown(create_download_link(video_path), unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"حدث خطأ أثناء إنشاء الفيديو: {str(e)}")

if __name__ == "__main__":
    main()