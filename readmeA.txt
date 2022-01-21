Αρβανιτέλλης Βάιος Ιάσων 1800010	
Πουρνάρας Γεώργιος 1800162
To πρόγραμμα φτιάχτηκε σε google colab.

A)
Για predict σε αρχείο με προυπάρχον μοντέλο από το Atrain.py
python forecast.py -d ./example.csv -n 9 oldmodel

Για δημιουργία νέου μοντέλου σε αρχείο και train,predict σε αυτό
python forecast.py -d ./example.csv -n 9 newmodel

Για δημιουργία και σώσιμο μοντέλου χωρίς να γίνουν προβλέψει σε αρχείο:
python Atrain.py ./example.csv 

Aκολουθήσαμε το άρθρο που δόθηκε.
Καταλήξαμε σε:window=60,epochs=5
To Αtrain εκπαιδεύει το μοντέλο στο 80% της κάθε μετοχής.