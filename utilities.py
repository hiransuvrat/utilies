
from csv import DictReader

def csvToVw(csvFile, outFile, numericalList, categoricalList, namespace, label, idCol, isTrain):

  with open(outFile,"wb") as outfile:
    for e, row in enumerate( DictReader(open(csvFile)) ):
      features = ""
      for k,v in row.items():
        if len(v) > 0:
          try:
            name = namespace[k]
          except KeyError:
            name = 'z'
          if k in categoricalList:
            features += " |%s %s_%s" % (name, k, v)
          elif k in numericalList:
            features += " |%s %s:%s" % (name, k, v)

      if isTrain: 
        outfile.write( "%s '%s %s\n" % (row[label], row[idCol],features) )

      else: 
        outfile.write( "1 '%s %s\n" % (row[idCol],features) )
      
      if e % 1000000 == 0:
        print("%s"%(e))

def csvToLibSVM(csvFile, outFile, numericalList, categoricalList, label, isTrain):
  with open(outFile,"wb") as outfile:
    for e, row in enumerate( DictReader(open(csvFile)) ):
      features = ""
      for k,v in row.items():
        if len(v) > 0:
          hashValue = (hash("%s_%s" % (k, v)) % 2000000000)
          if k in categoricalList:
            features += " %s:1" % (hashValue)
          elif k in numericalList:
            features += " %s:%s" % (hashValue, v)

      if isTrain: 
        outfile.write( "%s%s\n" % (row[label], features) )

      else: 
        outfile.write( "1%s\n" % (features))
      
      if e % 1000000 == 0:
        print("%s"%(e))

#csvToVw("test.csv", "test.vw", ['I1','I2','I3','I4','I5','I6','I7','I8','I9','I10','I11','I12','I13'],['C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14','C15','C16','C17','C18','C19','C20','C21','C22','C23','C24','C25','C26'],{'I1':'i','I2':'i','I3':'i','I4':'i','I5':'i','I6':'i','I7':'i','I8':'i','I9':'i','I10':'i','I11':'i','I12':'i','I13':'i','C1':'c','C2':'c','C3':'c','C4':'c','C5':'c','C6':'c','C7':'c','C8':'c','C9':'c','C10':'c','C11':'c','C12':'c'}, "Label", "Id", True)

csvToLibSVM("test.csv", "test.svm", ['I1','I2','I3','I4','I5','I6','I7','I8','I9','I10','I11','I12','I13'],['C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14','C15','C16','C17','C18','C19','C20','C21','C22','C23','C24','C25','C26'], "Label", True)
