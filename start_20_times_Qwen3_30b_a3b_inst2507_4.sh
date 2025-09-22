for i in {1..20}
do
  count=0
  while true
  do
    python3 main_2.py --config settings_api_qwen3-30b-a3b-inst2507_4.yaml
    if [ $? -eq 0 ]; then
      break
    fi
    count=$((count+1))
    if [ $count -ge 10 ]; then
      echo "10回リトライしても失敗しました。"
      break
    fi
    echo "エラーが発生したためリトライします ($count/10)"
    sleep 1
  done
done