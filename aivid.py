import streamlit as st
from moviepy.editor import VideoFileClip, TextClip, CompositeVideoClip
import torch
from diffusers import TextToVideoSDPipeline, DPMSolverMultistepScheduler
import tempfile
import os
from PIL import Image
import base64
import numpy as np

# تعيين إعدادات الصفحة
st.set_page_config(
    page_title="منشئ الفيديوهات بالذكاء الاصطناعي",
    page_icon="🎥",
    layout="wide"
)

@st.cache_resource
def load_model():
    """تحميل النموذج مع التخزين المؤقت"""
    pipe = TextToVideoSDPipeline.from_pretrained(
        "damo-vilab/text-to-video-ms-1.7b",
        torch_dtype=torch.float16,
        variant="fp16"
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    
    if torch.cuda.is_available():
        pipe = pipe.to("cuda")
    else:
        pipe = pipe.to("cpu")
    
    return pipe

def create_download_link(video_path):
    """إنشاء رابط لتحميل الفيديو"""
    with open(video_path, 'rb') as f:
        bytes = f.read()
        b64 = base64.b64encode(bytes).decode()
        href = f'<a href="data:file/mp4;base64,{b64}" download="generated_video.mp4">اضغط هنا لتحميل الفيديو</a>'
        return href

def generate_video(pipe, prompt, duration=5):
    """توليد فيديو باستخدام نموذج الذكاء الاصطناعي"""
    num_frames = duration * 8  # 8 FPS
    
    # توليد الفيديو
    video_frames = pipe(
        prompt,
        num_inference_steps=25,
        num_frames=num_frames,
        height=256,  # تقليل حجم الإطار للأداء الأفضل
        width=256
    ).frames
    
    # تحويل الإطارات إلى مصفوفة NumPy
    video_frames_np = np.array([np.array(frame) for frame in video_frames])
    
    # حفظ الفيديو مؤقتاً
    temp_dir = tempfile.mkdtemp()
    output_path = os.path.join(temp_dir, "generated_video.mp4")
    
    # إنشاء VideoFileClip من الإطارات
    clip = VideoFileClip(video_frames_np, fps=8)
    clip.write_videofile(output_path, codec='libx264', fps=8)
    
    return output_path

def main():
    st.title("🎥 منشئ الفيديوهات بالذكاء الاصطناعي")
    
    # إضافة شريط تقدم التحميل
    with st.spinner('جاري تحميل النموذج... (قد يستغرق هذا بضع دقائق في المرة الأولى)'):
        try:
            pipe = load_model()
            st.success('تم تحميل النموذج بنجاح!')
        except Exception as e:
            st.error(f"خطأ في تحميل النموذج: {str(e)}")
            return
    
    # إدخال النص الوصفي
    prompt = st.text_input(
        "أدخل وصفاً للفيديو الذي تريد إنشاءه",
        "مشهد لغروب الشمس على الشاطئ"
    )
    
    col1, col2 = st.columns(2)
    
    with col1:
        duration = st.slider("مدة الفيديو (بالثواني)", 3, 10, 5)
    
    with col2:
        quality = st.select_slider(
            "جودة الفيديو",
            options=["منخفضة", "متوسطة", "عالية"],
            value="متوسطة"
        )
    
    if st.button("إنشاء الفيديو", type="primary"):
        with st.spinner('جاري إنشاء الفيديو... يرجى الانتظار'):
            try:
                video_path = generate_video(pipe, prompt, duration)
                
                # عرض الفيديو
                st.video(video_path)
                
                # إضافة زر التحميل
                st.markdown(create_download_link(video_path), unsafe_allow_html=True)
                
            except Exception as e:
                st.error(f"حدث خطأ أثناء إنشاء الفيديو: {str(e)}")
                st.info("نصيحة: حاول استخدام وصف أبسط أو تقليل مدة الفيديو")

if __name__ == "__main__":
    main()