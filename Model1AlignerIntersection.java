package edu.berkeley.nlp.assignments.align.student;

import edu.berkeley.nlp.langmodel.EnglishWordIndexer;
import edu.berkeley.nlp.mt.Alignment;
import edu.berkeley.nlp.mt.SentencePair;
import edu.berkeley.nlp.mt.WordAligner;
import edu.berkeley.nlp.util.CounterMap;
import edu.berkeley.nlp.util.Indexer;
import edu.berkeley.nlp.util.StringIndexer;

/**
 * Created by aaksha on 12/1/16.
 */
public class Model1AlignerIntersection implements WordAligner {
    StringIndexer englishWords= EnglishWordIndexer.getIndexer();
    Indexer<String> frenchWords=new Indexer<>();
    CounterMap f2epairCounter =new CounterMap<Integer,Integer>();
    CounterMap e2fpairCounter =new CounterMap<Integer,Integer>();
    int em_iters=5;
    double nullposprob=0.2;

    public Alignment alignSentencePair(SentencePair sentencePair){
        int[] alignmente2f=getAlignmente2f(sentencePair);
        int[] alignmentf2e=getAlignmentf2e(sentencePair);
        Alignment a=new Alignment();
        for(int i=0;i<alignmentf2e.length;i++){
            if(alignmentf2e[i]!=-1) {
                int englishpos=alignmentf2e[i];
                if(alignmente2f[englishpos]==i) a.addAlignment(alignmentf2e[i], i, true);
            }
        }
        return a;
    }
    public void train(Iterable<SentencePair> trainingSentencePairs){
        trainf2e(trainingSentencePairs);
        traine2f(trainingSentencePairs);
    }

    public void trainf2e(Iterable<SentencePair> trainingSentencePairs) {
        initializeCountersf2e(trainingSentencePairs);

        for(int iter=0;iter<em_iters;iter++){
            CounterMap<Integer,Integer> newpairCounter=new CounterMap<>();
            for(SentencePair esfs: trainingSentencePairs){
                int esl = esfs.getEnglishWords().size();
                int fsl = esfs.getFrenchWords().size();
                int[] alignment=getAlignmentf2e(esfs);
                for(int i=0;i<fsl;i++){
                    String fw=esfs.getFrenchWords().get(i);
                    int fi=frenchWords.addAndGetIndex(fw);
                    if(alignment[i]==-1){
                        newpairCounter.incrementCount(-1,fi,1);
                    }
                    else{
                        String ew=esfs.getEnglishWords().get(alignment[i]);
                        int ei=englishWords.addAndGetIndex(ew);
                        newpairCounter.incrementCount(ei,fi,1);
                    }
                }
            }
            newpairCounter.normalize();
            f2epairCounter =newpairCounter;
        }
    }
    public void traine2f(Iterable<SentencePair> trainingSentencePairs) {
        initializeCounterse2f(trainingSentencePairs);

        for(int iter=0;iter<em_iters;iter++){
            System.out.println("Iter no.:"+Integer.toString(iter));
            CounterMap<Integer,Integer> newpairCounter=new CounterMap<>();
            for(SentencePair esfs: trainingSentencePairs){
                int esl = esfs.getEnglishWords().size();
                int fsl = esfs.getFrenchWords().size();
                int[] alignment=getAlignmente2f(esfs);
                for(int i=0;i<esl;i++){
                    String ew=esfs.getEnglishWords().get(i);
                    int ei=englishWords.addAndGetIndex(ew);
                    if(alignment[i]==-1){
                        newpairCounter.incrementCount(-1,ei,1);
                    }
                    else{
                        String fw=esfs.getFrenchWords().get(alignment[i]);
                        int fi=frenchWords.addAndGetIndex(fw);
                        newpairCounter.incrementCount(fi,ei,1);
                    }
                }
            }
            newpairCounter.normalize();
            e2fpairCounter =newpairCounter;
        }
    }
    public int[] getAlignmentf2e(SentencePair esfs){
        int esl = esfs.getEnglishWords().size();
        int fsl = esfs.getFrenchWords().size();
        int[] alignment=new int[fsl];
        for(int i=0;i<fsl;i++){
            String fw=esfs.getFrenchWords().get(i);
            int fi=frenchWords.addAndGetIndex(fw);

            double bestprob=nullposprob* f2epairCounter.getCount(-1,fi);
            alignment[i]=-1;

            for(int j=0;j<esl;j++){
                String ew=esfs.getEnglishWords().get(j);
                int ei=englishWords.addAndGetIndex(ew);
                double prob=getDistortionProb(esl)* f2epairCounter.getCount(ei,fi);
                if(prob>bestprob){
                    bestprob=prob;
                    alignment[i]=j;
                }
            }

        }
        return alignment;
    }
    public int[] getAlignmente2f(SentencePair esfs){
        int esl = esfs.getEnglishWords().size();
        int fsl = esfs.getFrenchWords().size();
        int[] alignment=new int[esl];
        for(int i=0;i<esl;i++){
            String ew=esfs.getEnglishWords().get(i);
            int ei=englishWords.addAndGetIndex(ew);

            double bestprob=nullposprob* e2fpairCounter.getCount(-1,ei);
            alignment[i]=-1;

            for(int j=0;j<fsl;j++){
                String fw=esfs.getFrenchWords().get(j);
                int fi=frenchWords.addAndGetIndex(fw);
                double prob=getDistortionProb(fsl)* e2fpairCounter.getCount(fi,ei);
                if(prob>bestprob){
                    bestprob=prob;
                    alignment[i]=j;
                }
            }

        }
        return alignment;
    }
    public double getDistortionProb(int sl){
        double prob=(1.0-nullposprob)/(sl+1);
        return prob;
    }
    public void initializeCountersf2e(Iterable<SentencePair> trainingSentencePairs){
        for (SentencePair esfs: trainingSentencePairs) {
            int esl = esfs.getEnglishWords().size();
            int fsl = esfs.getFrenchWords().size();
            for(int i=0;i<fsl;i++){
                String fw=esfs.getFrenchWords().get(i);
                int fi=frenchWords.addAndGetIndex(fw);
                f2epairCounter.setCount(-1,fi,1);
                for(int j=0;j<esl;j++){
                    String ew=esfs.getEnglishWords().get(j);
                    int ei=englishWords.addAndGetIndex(ew);
                    f2epairCounter.incrementCount(ei,fi,1);
                }
            }
        }
        f2epairCounter.normalize();
    }
    public void initializeCounterse2f(Iterable<SentencePair> trainingSentencePairs){
        for (SentencePair esfs: trainingSentencePairs) {
            int esl = esfs.getEnglishWords().size();
            int fsl = esfs.getFrenchWords().size();
            for(int i=0;i<esl;i++){
                String ew=esfs.getEnglishWords().get(i);
                int ei=englishWords.addAndGetIndex(ew);
                e2fpairCounter.setCount(-1,ei,1);
                for(int j=0;j<fsl;j++){
                    String fw=esfs.getFrenchWords().get(j);
                    int fi=frenchWords.addAndGetIndex(fw);
                    e2fpairCounter.incrementCount(fi,ei,1);
                }
            }
        }
        e2fpairCounter.normalize();
    }
}
