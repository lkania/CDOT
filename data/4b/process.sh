tail -n +2 Last3Bins3b.txt | sed 's/.*"//' | sed 's/^[ \t]*//' > m_muamu.txt
tail -n +2 Last3BinsSignal.txt | sed 's/.*"//' | sed 's/^[ \t]*//' > signal.txt
