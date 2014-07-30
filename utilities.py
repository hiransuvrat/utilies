from csv import DictReader


#Converts csv file to vw format
def csvToVw(csvFile, outFile, numericalList, categoricalList, namespace, label, idCol, isTrain, smoothing = [], smoothingPrefix = 'smoothies'):
  smoothFields = {}
  for fields in smoothing:
    smoothFields[fields] = {}
    for e, line in enumerate( open("%s_%s" % (smoothingPrefix, fields) ) ):
      val = line.split(",")
      smoothFields[fields][val[0]] = val[1]

  print 'Read smoothing fields'
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
          elif k in smoothing:
            try:
              features += " |%s %s:%s" % (name, k, smoothFields[k][v].strip())
            except KeyError:
              features += " |%s %s:.25" % (name, k)

      if isTrain: 
        if row[label] == '0':
          row[label] = '-1'
        outfile.write( "%s '%s %s\n" % (row[label], row[idCol],features) )
      else: 
        outfile.write( "1 '%s %s\n" % (row[idCol],features) )
      
      if e % 1000000 == 0:
        print("%s"%(e))

#Converts csv to libsvm format
def csvToLibSVM(csvFile, outFile, numericalList, categoricalList, label, isTrain, smoothing = [], smoothingPrefix = 'smoothies'):
  smoothFields = {}
  for fields in smoothing:
    smoothFields[fields] = {}
    for e, line in enumerate( open("%s_%s" % (smoothingPrefix, fields) ) ):
      val = line.split(",")
      smoothFields[fields][val[0]] = val[1]

  print 'Read smoothing fields'

  hashAt = 5000
  with open(outFile,"wb") as outfile:
    for e, row in enumerate( DictReader(open(csvFile)) ):
      features = {}
      for k,v in row.items():
        if len(v) > 0:
          if k in categoricalList:
            hashValue = (hash("%s_%s" % (k, v)) % hashAt)
            features[hashValue] = '1'
          elif k in numericalList:
            hashValue = (hash("%s" % (k)) % hashAt)
            features[hashValue] = v
          elif k in smoothing:
            try:
              hashValue = (hash("%s" % (k)) % hashAt)
              features[hashValue] = smoothFields[k][v].strip()
            except KeyError:
              hashValue = (hash("%s" % (k)) % hashAt)
              features[hashValue] = '.25'
      featureVector = ''
      for (key, value) in sorted(features.items()):
        featureVector += " %s:%s" % (key, value)
      if isTrain: 
        outfile.write( "%s%s\n" % (row[label], featureVector) )

      else: 
        outfile.write( "1%s\n" % (featureVector))
      
      if e % 1000000 == 0:
        print("%s"%(e))

#Dumps categorical variables to smoothed values
def dumpSmoothies(csvFile, outFilePrefix, fields, alpha = 300, beta = 75, label = 'Label'):
  dictFields = {}

  #Initialize dictionaries
  for i in fields:
    dictFields[i] = {}

  for e, row in enumerate( DictReader(open(csvFile)) ):  
    for k,v in row.items():
      if k in fields:
        try:
          if row['Label'] == '0':
            dictFields[k][v][1] += 1 
          else:
            dictFields[k][v][0] += 1 
        except KeyError:
          dictFields[k][v] = {}
          dictFields[k][v] = {}
          if row['Label'] == '1':
            dictFields[k][v][0] = beta + 1
            dictFields[k][v][1] = alpha + 1
          else:
            dictFields[k][v][0] = beta 
            dictFields[k][v][1] = alpha + 1
    if e % 1000000 == 0:
      print("%s"%(e))

  print "Smoothies on the table"

  for i in fields:
    with open("%s_%s" % (outFilePrefix, i),"wb") as outfile:
      outfile.write( "%s,val\n" % (i))
      for key in dictFields[i]:
        avgVal = float(dictFields[i][key][0]) / float(dictFields[i][key][1])
        outfile.write( "%s,%s\n" % (key,str(avgVal)))
    print "%s flavour for you" % (i)

#csvToVw("test.csv", "test.vw", ['I1','I2','I3','I4','I5','I6','I7','I8','I9','I10','I11','I12','I13'],['C1','C2','C3','C4','C5','C6','C7','C8','C9','C10','C11','C12','C13','C14','C15','C16','C17','C18','C19','C20','C21','C22','C23','C24','C25','C26'],{'I1':'i','I2':'i','I3':'i','I4':'i','I5':'i','I6':'i','I7':'i','I8':'i','I9':'i','I10':'i','I11':'i','I12':'i','I13':'i','C1':'c','C2':'c','C3':'c','C4':'c','C5':'c','C6':'c','C7':'c','C8':'c','C9':'c','C10':'c','C11':'c','C12':'c'}, "Label", "Id", True)

csvToVw("/mnt/crit/train2.csv", "/mnt/crit/train2.vw", ['I1','I2','I3','I4','I5','I6','I7','I8','I9','I10','I11','I12','I13', 'I14'],['C6','C9','C14', 'C17','C20', 'C22','C23','C25'],{'I1':'i','I2':'i','I3':'i','I4':'i','I5':'i','I6':'i','I7':'i','I8':'i','I9':'i','I10':'i','I11':'i','I12':'i','I13':'i','C1':'c','C2':'c','C3':'c','C4':'c','C5':'c','C6':'j','C7':'c','C8':'c','C9':'k','C10':'c','C11':'c','C12':'c', 'C14':'l', 'C20':'e', 'C23':'f', 'C25':'g', 'I14':'h'}, "Label", "Id", True, ['C1', 'C2', 'C3','C4', 'C5', 'C7', 'C8', 'C10', 'C11', 'C12', 'C13', 'C15', 'C16', 'C18', 'C19', 'C21','C24', 'C26'])

#csvToVw("test.csv", "test.vw", ['I1','I2','I3','I4','I5','I6','I7','I8','I9','I10','I11','I12','I13'],['C6','C9','C14', 'C17','C20', 'C22','C23','C25'],{'I1':'i','I2':'i','I3':'i','I4':'i','I5':'i','I6':'i','I7':'i','I8':'i','I9':'i','I10':'i','I11':'i','I12':'i','I13':'i','C1':'c','C2':'c','C3':'c','C4':'c','C5':'c','C6':'a','C7':'c','C8':'c','C9':'b','C10':'c','C11':'c','C12':'c', 'C14':'d', 'C20':'e', 'C23':'f', 'C25':'g'}, "Label", "Id", True, ['C1', 'C2', 'C3','C4', 'C5', 'C7', 'C8', 'C10', 'C11', 'C12', 'C13', 'C15', 'C16', 'C18', 'C19', 'C21','C24', 'C26'])

#csvToLibSVM("/mnt/crit/train2.csv", "/mnt/crit/train2.svm", ['I1','I2','I3','I4','I5','I6','I7','I8','I9','I10','I11','I12','I13'],['C1','C2','C5','C6','C8','C9','C13','C14','C17','C18','C19','C20','C22','C23','C25'], "Label", True, ['C3','C4','C10','C12','C16','C21','C24', 'C1', 'C2', 'C5', 'C7', 'C8', 'C11','C13', 'C15', 'C17', 'C18', 'C19', 'C26'])

#dumpSmoothies("/mnt/crit/train.csv", "smoothies", ['C3', 'C4', 'C10', 'C12', 'C16', 'C21','C24', 'C26'])

#dumpSmoothies("/mnt/crit/train.csv", "smoothies", ['C1', 'C2', 'C5', 'C7', 'C8', 'C11','C13', 'C15', 'C17', 'C18', 'C19', 'C26'])
