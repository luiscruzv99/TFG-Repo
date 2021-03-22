for i in `seq 1 10`
do
  echo "=========Run $i===========" >> result_batch.txt;
  echo "TEMPS: " >> result_batch.txt;
  sensors >> result_batch.txt;
  echo "TIMES: " >> result_batch.txt;
  python3 test_mark.py >> result_batch.txt;
  echo "TEMPS: " >> result_batch.txt;
  sensors >> result_batch.txt;
done
