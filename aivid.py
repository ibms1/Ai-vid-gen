import streamlit as st
from moviepy.editor import VideoFileClip, concatenate_videoclips
import torch
from diffusers import TextToVideoSDPipeline
import tempfile
import os
import numpy as np
from PIL import Image
import gc

# تعيين إعدادات الصفحة
st.set_page_config(
    page_title="منشئ الفيديوهات السريع",
    page_icon="🎥",
    layout="wide"
)

# تحسين استخدام الذاكرة
@st.cache_resource
def load_model():
    """تحميل النموذج مع تحسينات الأداء"""
    pipe = TextToVideoSDPipeline.from_pretrained(
        "cerspense/zeroscope_v2_576w",  # نموذج أخف وأسرع
        torch_dtype=torch.float16
    )
    
    if torch.cuda.is_available():
        pipe.enable_model_cpu_offload()  # تحسين استخدام الذاكرة
        pipe.enable_vae_slicing()  # تحسين أداء VAE
    else:
        pipe = pipe.to("cpu")
    
    return pipe

def clear_memory():
    """تنظيف الذاكرة"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

def generate_video(pipe, prompt, num_frames=24):
    """توليد فيديو مع إعدادات محسنة"""
    try:
        # إعدادات محسنة لتوليد أسرع
        video_frames = pipe(
            prompt,
            num_inference_steps=20,  # تقليل خطوات الاستدلال
            num_frames=num_frames,
            height=320,  # تقليل الحجم للسرعة
            width=576,
            guidance_scale=7.0  # تقليل مقياس التوجيه
        ).frames
        
        # تحويل الإطارات إلى فيديو
        temp_dir = tempfile.mkdtemp()
        output_path = os.path.join(temp_dir, "generated_video.mp4")
        
        # حفظ كإطارات مؤقتة
        frames_array = np.array([np.array(frame) for frame in video_frames])
        
        # إنشاء الفيديو بإعدادات محسنة
        clip = VideoFileClip(frames_array, fps=8)
        clip.write_videofile(
            output_path,
            fps=8,
            codec='libx264',
            preset='ultrafast',
            threads=4
        )
        
        clear_memory()  # تنظيف الذاكرة بعد التوليد
        return output_path
        
    except Exception as e:
        raise Exception(f"خطأ في توليد الفيديو: {str(e)}")

def main():
    st.title("🎥 منشئ الفيديوهات السريع")
    
    # إضافة معلومات مفيدة
    st.info("""
    نصائح للحصول على أفضل النتائج:
    - استخدم وصفاً قصيراً وواضحاً
    - اختر المدة القصيرة للحصول على نتائج أسرع
    - جرب الوضع السريع أولاً قبل استخدام الجودة العالية
    """)
    
    # تحميل النموذج
    with st.spinner('جاري تحميل النموذج...'):
        try:
            pipe = load_model()
            st.success('✅ تم تحميل النموذج بنجاح!')
        except Exception as e:
            st.error(f"خطأ في تحميل النموذج: {str(e)}")
            return
    
    # إعدادات توليد الفيديو
    with st.form("video_generation_form"):
        prompt = st.text_input(
            "وصف الفيديو",
            placeholder="مثال: غروب الشمس على الشاطئ"
        )
        
        col1, col2 = st.columns(2)
        
        with col1:
            mode = st.radio(
                "وضع التوليد",
                options=["سريع", "عالي الجودة"],
                index=0
            )
        
        with col2:
            if mode == "سريع":
                num_frames = st.slider("عدد الإطارات", 16, 24, 20)
            else:
                num_frames = st.slider("عدد الإطارات", 24, 32, 28)
        
        submitted = st.form_submit_button("إنشاء الفيديو")
    
    if submitted and prompt:
        try:
            # إظهار شريط التقدم
            progress_text = "جاري إنشاء الفيديو..."
            progress_bar = st.progress(0)
            
            # توليد الفيديو مع تحديث التقدم
            for i in range(100):
                progress_bar.progress(i + 1)
                if i == 20:
                    st.info("جاري معالجة الوصف...")
                elif i == 40:
                    st.info("جاري توليد الإطارات...")
                elif i == 60:
                    st.info("جاري دمج الإطارات...")
                elif i == 80:
                    st.info("جاري تحسين الفيديو...")
            
            video_path = generate_video(pipe, prompt, num_frames)
            
            # عرض الفيديو
            st.video(video_path)
            
            # إضافة زر التحميل
            with open(video_path, 'rb') as f:
                st.download_button(
                    label="تحميل الفيديو",
                    data=f,
                    file_name="generated_video.mp4",
                    mime="video/mp4"
                )
            
            # تنظيف الذاكرة
            clear_memory()
            
        except Exception as e:
            st.error(f"حدث خطأ: {str(e)}")
            st.info("حاول مرة أخرى مع وصف أبسط أو عدد إطارات أقل")

if __name__ == "__main__":
    main()