import gradio as gr
from main import load_model, generate_summary, model, tokenizer, device
import yt_dlp as youtube_dl
from faster_whisper import WhisperModel
import os
import argparse

def download_video(video_url):
    ydl_opts = {
        'format': 'bestaudio/best',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'mp3',
            'preferredquality': '192',
        }],
        'outtmpl': 'downloaded_audio.%(ext)s',
        'quiet': True,
        'no_warnings': True,
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(video_url, download=True)
        title = info.get("title", "Unknown Title")
    return "downloaded_audio.mp3", title

def transcribe_audio(audio_path):
    whisper_model = WhisperModel("base")
    segments, _ = whisper_model.transcribe(audio_path)
    transcription = " ".join([segment.text for segment in segments])
    return transcription

def process_video(video_url):
    # Step 1: Download and get title
    audio_path, title = download_video(video_url)
    
    # Step 2: Transcribe
    transcription = transcribe_audio(audio_path)
    
    # Step 3: Clean up
    if os.path.exists(audio_path):
        os.remove(audio_path)
    
    # Step 4: Generate summary
    summary = generate_summary(model, tokenizer, device, transcription, model_type=args.model_type)
    
    return title, transcription, summary

def video_summarization_interface():
    with gr.Blocks() as demo:
        gr.Markdown(f"# Video Summarization")
        video_url = gr.Textbox(label="YouTube Video URL")
        
        video_title = gr.Textbox(label="Video Title", interactive=False)
        transcription_output = gr.Textbox(label="Transcription", lines=10, interactive=False)
        summary_output = gr.Textbox(label="Summary", lines=10, interactive=False)
        
        process_button = gr.Button("Process Video")
        
        # Display title first, then transcription, then summary
        process_button.click(
            process_video, 
            inputs=video_url, 
            outputs=[video_title, transcription_output, summary_output]
        )
    return demo

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, choices=['t5', 'bart'], default='t5')
    args = parser.parse_args()
    
    if args.model_type == 't5':
        model_path = f'models/t5_summarizer_epoch_15_full_data.pt'
    else:
        model_path = f'models/bart_model_epoch_8.pt'

    # Load model, tokenizer, and device here (uncomment this line)
    # model, tokenizer, device = load_model(model_path, model_type=args.model_type)

    demo = video_summarization_interface()
    demo.launch()