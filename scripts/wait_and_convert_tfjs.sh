#!/bin/bash
# tensorflowjs 설치 완료 대기 및 자동 변환

MODEL="model.h5"
OUTPUT="public/model"
CHECK_INTERVAL=60  # 60초마다 확인
MAX_WAIT=1800      # 최대 30분 대기

echo "🔍 tensorflowjs 설치 완료를 확인 중..."
echo "   확인 간격: ${CHECK_INTERVAL}초"
echo "   최대 대기: ${MAX_WAIT}초 (30분)"
echo ""

elapsed=0
while [ $elapsed -lt $MAX_WAIT ]; do
    if pip3 show tensorflowjs > /dev/null 2>&1; then
        echo ""
        echo "✅ tensorflowjs 설치 완료! (소요 시간: $((elapsed / 60))분 $((elapsed % 60))초)"
        echo ""
        echo "🔄 모델 변환 시작..."
        python3 scripts/convert_to_tfjs.py --model "$MODEL" --output "$OUTPUT"
        
        if [ $? -eq 0 ]; then
            echo ""
            echo "🎉 모든 작업 완료!"
            echo "📁 변환된 모델 위치: $OUTPUT"
            echo ""
            echo "생성된 파일:"
            ls -lh "$OUTPUT"/
            exit 0
        else
            echo "❌ 변환 중 오류 발생"
            exit 1
        fi
    fi
    
    echo -ne "\r⏳ 설치 대기 중... ($(date +%H:%M:%S)) - $((elapsed / 60))분 $((elapsed % 60))초 경과"
    sleep $CHECK_INTERVAL
    elapsed=$((elapsed + CHECK_INTERVAL))
done

echo ""
echo "⏰ 최대 대기 시간 초과"
exit 1

