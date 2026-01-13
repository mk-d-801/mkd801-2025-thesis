python MELD/teacher.py | tee -a log/MELD_log.txt
python MELD/student.py | tee -a log/MELD_log.txt
python MELD/fusion.py | tee -a log/MELD_log.txt
python MELD/inference.py | tee -a log/MELD_log.txt

python MELD/tsne.py | tee -a log/MELD_log.txt
python MELD/wc.py | tee -a log/MELD_log.txt