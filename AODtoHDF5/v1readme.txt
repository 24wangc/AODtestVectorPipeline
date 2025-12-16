[Start HDF5er(vbfBool, signalBool, jzSlice)]
           |
           v
   [Get input directory for this sample]
           |
           v
   [Scan directory]
   [Find DAOD_JETM42*.root files]
           |
           v
   [If no files → print error and return]
           |
           v
   [Initialize big vectors for:
      - Event info (weights)
      - Truth Higgs/b-quarks
      - Towers & clusters
      - L1 jets (gFEX, jFEX)
      - HLT jets
      - reco AK10 jets
      - truth jets (WZ, in-time)
      - Jagged offsets
   ]
           |
           v
   [Start stopwatch for event processing]
           |
           v
   ┌───────────────────────────────────────────────┐
   │         For each ROOT file in fileNames      │
   │    (open file, wrap in xAOD::TEvent)         │
   └───────────────────────────────────────────────┘
           |
           v
   ┌───────────────────────────────────────────────┐
   │           For each event in file             │
   │       (up to 1000 events per file)           │
   └───────────────────────────────────────────────┘
           |
           v
   [Apply filters and, if event is good,
    retrieve containers and fill vectors]
           |
           v
   [After all files: stop stopwatch,
    print timing + counters]
           |
           v
   [Open exampleHDF5.h5, create groups]
           |
           v
   [Write all vectors to HDF5 datasets]
           |
           v
   [Print write timing]
           |
           v
          [End]
