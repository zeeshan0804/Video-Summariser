import gradio as gr
from main import load_model, generate_summary
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
    }
    with youtube_dl.YoutubeDL(ydl_opts) as ydl:
        ydl.download([video_url])
    return "downloaded_audio.mp3"

def transcribe_audio(audio_path):
    whisper_model = WhisperModel("base")
    segments, _ = whisper_model.transcribe(audio_path)
    transcription = " ".join([segment.text for segment in segments])
    return transcription

def process_video(video_url):
    # Step 1: Download
    audio_path = download_video(video_url)
    
    # Step 2: Transcribe
    transcription = transcribe_audio(audio_path)
    
    # Clean up
    if os.path.exists(audio_path):
        os.remove(audio_path)
    
    summary = generate_summary(model, tokenizer, device, transcription, model_type=args.model_type)
    return transcription, summary

def video_summarization_interface():
    with gr.Blocks() as demo:
        gr.Markdown(f"# Video Summarization (Using {args.model_type.upper()} Model)")
        video_url = gr.Textbox(label="YouTube Video URL")
        with gr.Row():
            transcription_output = gr.Textbox(label="Transcription", lines=10)
            summary_output = gr.Textbox(label="Summary", lines=10)
        process_button = gr.Button("Process Video")
        process_button.click(process_video, inputs=video_url, outputs=[transcription_output, summary_output])
    return demo

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_type', type=str, choices=['t5', 'bart'], default='t5')
    args = parser.parse_args()
    
    if args.model_type == 't5':
        model_path = f'models/t5_summarizer_epoch_15_full_data.pt'
    else:
        model_path = f'models/bart_model_epoch_8.pt'

    model, tokenizer, device = load_model(model_path, model_type=args.model_type)

    demo = video_summarization_interface()
    demo.launch()