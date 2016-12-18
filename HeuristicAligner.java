package edu.berkeley.nlp.assignments.align.student;

import edu.berkeley.nlp.langmodel.EnglishWordIndexer;
import edu.berkeley.nlp.math.DoubleArrays;
import edu.berkeley.nlp.mt.Alignment;
import edu.berkeley.nlp.mt.SentencePair;
import edu.berkeley.nlp.mt.WordAligner;
import edu.berkeley.nlp.util.*;

import java.util.List;

/**
 * Created by aaksha on 11/22/16.
 */
public class HeuristicAligner implements WordAligner {
    StringIndexer englishWords= EnglishWordIndexer.getIndexer();
    IntCounter englishCounter=new IntCounter();
    Indexer<String> frenchWords=new Indexer<>();
    IntCounter frenchCounter=new IntCounter();
    Indexer<String> pairs=new Indexer<>();
    IntCounter pairCounter=new IntCounter();


    public Alignment alignSentencePair(SentencePair sentencePair) {
        Alignment alignment = new Alignment();
        int esl=sentencePair.getEnglishWords().size();
        int fsl=sentencePair.getFrenchWords().size();

        for (int i = 0; i < esl; i++) {
            String ew = sentencePair.getEnglishWords().get(i);
            if(englishWords.contains(ew)){
                int ei=englishWords.addAndGetIndex(ew);
                int bestfw=-1;
                double bestfwratio=0;
                for(int j=0;j<fsl;j++){
                    String fw=sentencePair.getFrenchWords().get(j);
                    if(frenchWords.contains(fw)){
                        int fi=frenchWords.addAndGetIndex(fw);
                        String pair=ew+"~"+fw;
                        int ipair=pairs.addAndGetIndex(pair);
                        double ratio=pairCounter.getCount(ipair)/(englishCounter.getCount(ei)*frenchCounter.getCount(fi));
                        if(ratio>bestfwratio){
                            bestfwratio=ratio;
                            bestfw=j;
                        }
                    }
                }
                if(bestfw!=-1) alignment.addAlignment(i,bestfw,true);
            }
        }
        return alignment;
    }
    public void train(Iterable<SentencePair> trainingSentencePairs) {
        for (SentencePair esfs: trainingSentencePairs){
            int esl=esfs.getEnglishWords().size();
            int fsl=esfs.getFrenchWords().size();
            for(int i=0;i<esl;i++){
                String ew=esfs.getEnglishWords().get(i);
                int ewi=englishWords.addAndGetIndex(ew);
                englishCounter.incrementCount(ewi,1);
            }

            for(int i=0;i<fsl;i++){
                String fw=esfs.getFrenchWords().get(i);
                int fwi=frenchWords.addAndGetIndex(fw);
                frenchCounter.incrementCount(fwi,1);
            }

            for(int i=0;i<esl;i++){
                for(int j=0;j<fsl;j++){
                    String ew=esfs.getEnglishWords().get(i);
                    String fw=esfs.getFrenchWords().get(j);
                    String pair=ew+"~"+fw;
                    int ipair=pairs.addAndGetIndex(pair);
                    pairCounter.incrementCount(ipair,1);
                }
            }
        }
        System.out.println("# English Words: " + englishWords.size());
        System.out.println("# French Words: " + frenchWords.size());
        System.out.println("Pairs: " + pairs.size());
        System.out.println("total pair count: " + pairCounter.totalCount());

    }

}
