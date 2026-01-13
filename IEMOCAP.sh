python IEMOCAP/teacher.py | tee -a log/IEMOCAP_log.txt
python IEMOCAP/student.py | tee -a log/IEMOCAP_log.txt
python IEMOCAP/fusion.py | tee -a log/IEMOCAP_log.txt
python IEMOCAP/inference.py | tee -a log/IEMOCAP_log.txt

python IEMOCAP/tsne.py | tee -a log/IEMOCAP_log.txt