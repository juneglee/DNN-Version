from pytube import YouTube
# 특정영상 다운로드
# golf_swing_1
# YouTube('https://www.youtube.com/watch?v=-lHQ2p-gdvE').streams.first().download()
# golf_swing_2
# YouTube('https://www.youtube.com/watch?v=Qh2U0AMzDMo').streams.first().download()
# 프리킥
# YouTube('https://www.youtube.com/watch?v=x9ENsNaZMaE').streams.first().download()
# 1초
# YouTube('https://www.youtube.com/watch?v=ZTlEDolI79I').streams.first().download()

# 야구 분석(피칭/모션캠)
# YouTube('https://www.youtube.com/watch?v=MFqnmF9VtZE').streams.first().download()
# yt = YouTube("https://www.youtube.com/watch?v=wd2Aj7y_WRA").streams.first().download()

# 배구
# YouTube('https://www.youtube.com/watch?v=Xw8U1NC2-Bk').streams.get_by_itag(137).download()

# 랜더링
#  https://www.youtube.com/watch?v=0ZnYRpGo7XU

# 의료
# https://www.youtube.com/watch?v=448bwGdxW04

# 영화
# https://www.youtube.com/watch?v=ijUsSpRVhBU

# 마블
# https://www.youtube.com/watch?v=sHmo2uGcP2Q
# https://www.youtube.com/watch?v=2VJXoVeGgzc

# 싸이
# https://www.youtube.com/watch?v=9bZkp7q19f0

# 침팬치
# https://www.youtube.com/watch?v=BGpaVv_kHZw

# 파이썬 소스코드
from pytube import YouTube
# 라이브러리 가져오기

yt = YouTube('https://www.youtube.com/watch?v=BGpaVv_kHZw')
# 동영상 링크를 이용해 YouTube 객체 생성

yt_streams = yt.streams
# YouTube 객체인 yt에서 stream 객체를 생성

print("다운가능한 영상 상세 정보 :")
for i, stream in enumerate(yt_streams.all()):
    print(i, " : ", stream)

print(" 고화질, mp4 포맷만 가져오기 : ")
for i, stream in enumerate(yt_streams.filter(adaptive=True, file_extension='mp4').all()):
    print(i, " : ", stream)

print(" \"itag\"를 이용해 특정 stream 선택 :")
itag = input()
my_stream = yt_streams.get_by_itag(itag)
print("선택된 stream : ", my_stream)

print(" 선택된 stream 다운로드 ")
my_stream.download()

