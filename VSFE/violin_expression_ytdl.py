from pydub import AudioSegment
import json
import youtube_dl
import os
import sys


def save_audio(_id):
    ydl_opts_audio = {
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                'preferredquality': '192',
            }],
            'postprocessor_args': [
                '-ar', '44100'
            ],
            'outtmpl': './violin_expression_youtube/audio/%(id)s.%(ext)s'
            }
    with youtube_dl.YoutubeDL(ydl_opts_audio) as ydl:
        ydl.download(['https://www.youtube.com/watch?v={}'.format(_id)])


def save_video(_id):
    ydl_opts_video = {
            'format': 'bestvideo[height<=720,ext=mp4]',
            'outtmpl': './violin_expression_youtube/video/%(id)s.%(ext)s'
            }
    with youtube_dl.YoutubeDL(ydl_opts_video) as ydl:
        ydl.download(['https://www.youtube.com/watch?v={}'.format(_id)])


if __name__ == '__main__':
    with open('./violin_yt_link_id') as f:
        ids = f.readlines()

    os.makedirs('./violin_expression_youtube/video', exist_ok=True)
    os.makedirs('./violin_expression_youtube/audio', exist_ok=True)

    for _id in ids:
        save_audio(_id)
        save_video(_id)
