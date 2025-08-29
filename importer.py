import os
from yt_dlp import YoutubeDL

def download_youtube_video(url, save_folder="vidDownloads"):
    os.makedirs(save_folder, exist_ok=True)
    ydl_opts = {
        'format': 'bestvideo+bestaudio/best',
        'outtmpl': os.path.join(save_folder, '%(title)s.%(ext)s'),
        'merge_output_format': 'mp4',
    }

    with YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        title = info.get('title', None)
        print(f"📥 Downloading: {title}")
        ydl.download([url])

        safe_title = title.replace(' ', '_').replace('/', '_').replace('\\', '_')
        output_path = os.path.join(save_folder, f"{safe_title}.mp4")
        print(f"✅ Download complete: {output_path}")
        return output_path

if __name__ == "__main__":
    url = input("🔗 Enter a YouTube URL to download: ")
    download_youtube_video(url)
