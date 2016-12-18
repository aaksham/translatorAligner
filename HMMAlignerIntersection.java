package edu.berkeley.nlp.assignments.align.student;

import edu.berkeley.nlp.langmodel.EnglishWordIndexer;
import edu.berkeley.nlp.mt.Alignment;
import edu.berkeley.nlp.mt.SentencePair;
import edu.berkeley.nlp.mt.WordAligner;
import edu.berkeley.nlp.util.*;

/**
 * Created by aaksha on 12/2/16.
 */
public class HMMAlignerIntersection implements WordAligner {
    StringIndexer englishWords= EnglishWordIndexer.getIndexer();
    Indexer<String> frenchWords=new Indexer<>();
    CounterMap f2epairCounter =new CounterMap<Integer,Integer>();
    CounterMap e2fpairCounter =new CounterMap<Integer,Integer>();

    int em_iters=5;
    double nullposprob=0.2;
    public Alignment alignSentencePair(SentencePair sentencePair){

        int esl = sentencePair.getEnglishWords().size();
        int fsl = sentencePair.getFrenchWords().size();

        double [][] gammaf2e=learnAlignmentf2e(sentencePair);
        int[] alignmentf2e=new int[fsl];
        for(int i=0;i<fsl;i++){
            int bestpos=esl;
            double bestgamma=gammaf2e[i][esl];
            for(int j=0;j<esl;j++){
                if (gammaf2e[i][j]>bestgamma){
                    bestgamma=gammaf2e[i][j];
                    bestpos=j;
                }
            }
            if(bestpos!=esl) alignmentf2e[i]=bestpos;
        }

        double [][] gammae2f=learnAlignmente2f(sentencePair);
        int[] alignmente2f=new int[esl];
        for(int i=0;i<esl;i++){
            int bestpos=fsl;
            double bestgamma=gammae2f[i][fsl];
            for(int j=0;j<fsl;j++){
                if (gammae2f[i][j]>bestgamma){
                    bestgamma=gammae2f[i][j];
                    bestpos=j;
                }
            }
            if(bestpos!=fsl) alignmente2f[i]=bestpos;
        }

        Alignment a = new Alignment();
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
    public void trainf2e(Iterable<SentencePair> trainingSentencePairs){
        initializeCountersf2e(trainingSentencePairs);
        System.out.println("French to English");
        for(int iter=0;iter<em_iters;iter++){
            System.out.println("Iter no.:"+Integer.toString(iter));
            CounterMap<Integer,Integer> newpairCounter=new CounterMap<>();
            for(SentencePair esfs: trainingSentencePairs){
                double [][] gamma=learnAlignmentf2e(esfs);
                int esl = esfs.getEnglishWords().size();
                int fsl = esfs.getFrenchWords().size();
                for(int i=0;i<fsl;i++){
                    String fw=esfs.getFrenchWords().get(i);
                    int fi=frenchWords.addAndGetIndex(fw);
                    for(int j=0;j<esl;j++){
                        String ew=esfs.getEnglishWords().get(j);
                        int ei=englishWords.addAndGetIndex(ew);
                        newpairCounter.incrementCount(ei,fi,gamma[i][j]);
                    }
                }
            }

            for(int englishi: newpairCounter.keySet()){
                Counter<Integer> marginal=newpairCounter.getCounter(englishi);
                if (marginal.totalCount()>0) marginal.normalize();
            }
            f2epairCounter =newpairCounter;
        }
    }
    public void traine2f(Iterable<SentencePair> trainingSentencePairs){
        initializeCounterse2f(trainingSentencePairs);
        System.out.println("English to French");
        for(int iter=0;iter<em_iters;iter++){
            System.out.println("Iter no.:"+Integer.toString(iter));
            CounterMap<Integer,Integer> newpairCounter=new CounterMap<>();
            for(SentencePair esfs: trainingSentencePairs){
                double [][] gamma=learnAlignmente2f(esfs);
                int esl = esfs.getEnglishWords().size();
                int fsl = esfs.getFrenchWords().size();
                for(int i=0;i<esl;i++){
                    String ew=esfs.getEnglishWords().get(i);
                    int ei=englishWords.addAndGetIndex(ew);
                    for(int j=0;j<fsl;j++){
                        String fw=esfs.getFrenchWords().get(j);
                        int fi=frenchWords.addAndGetIndex(fw);
                        newpairCounter.incrementCount(fi,ei,gamma[i][j]);
                    }
                }
            }

            for(int frenchi: newpairCounter.keySet()){
                Counter<Integer> marginal=newpairCounter.getCounter(frenchi);
                if (marginal.totalCount()>0) marginal.normalize();
            }
            e2fpairCounter =newpairCounter;
        }
    }
    public double[][] learnAlignmentf2e(SentencePair sentencePair){
        int esl = sentencePair.getEnglishWords().size();
        int fsl = sentencePair.getFrenchWords().size();

        double[][] alpha=new double[fsl][esl+1];
        double [][]gamma=new double[fsl][esl+1];

        double[][] transmat=new double[esl+1][esl+1];

        for(int i=0;i<esl;i++){
            transmat[i][esl]=nullposprob;
            transmat[esl][i]=(1-nullposprob)/esl;
            double normalization=0;
            for(int j=0;j<esl;j++){
                transmat[i][j]=Math.exp(-2*Math.abs(j-i-1.1));
                normalization+=transmat[i][j];
            }
            for(int j=0;j<esl;j++){
                transmat[i][j]=transmat[i][j]*(1-nullposprob)/normalization;
            }
        }
        String firstfw=sentencePair.getFrenchWords().get(0);
        int firstfi=frenchWords.addAndGetIndex(firstfw);

        double normalization=0;
        for(int i=0;i<esl;i++){
            String ew=sentencePair.getEnglishWords().get(i);
            int ei=englishWords.addAndGetIndex(ew);
            alpha[0][i]= f2epairCounter.getCount(ei,firstfi)*transmat[i][esl];
            normalization+=alpha[0][i];
        }
        alpha[0][esl]= f2epairCounter.getCount(-1,firstfi)*transmat[esl][esl];
        normalization+=alpha[0][esl];
        if(normalization!=0){
            for(int i=0;i<esl;i++) alpha[0][i]=alpha[0][i]/normalization;
        }

        for(int i=1;i<fsl;i++){
            String fw=sentencePair.getFrenchWords().get(i);
            int fi=frenchWords.addAndGetIndex(fw);

            normalization=0;
            for(int j=0;j<esl;j++){
                for(int k=0;k<esl;k++){
                    alpha[i][j]+=alpha[i-1][k]*transmat[k][j];
                }
                String ew=sentencePair.getEnglishWords().get(j);
                int ei=englishWords.addAndGetIndex(ew);
                alpha[i][j]*= f2epairCounter.getCount(ei,fi);
                normalization+=alpha[i][j];
            }
            if(normalization>0){
                for(int j=0;j<esl;j++) alpha[i][j]=alpha[i][j]/normalization;
            }
        }

        System.arraycopy(alpha[fsl-1],0,gamma[fsl-1],0,esl);
        double[] normdenoms=new double[esl+1];

        for(int i=fsl-2;i>=0;i--){
            for(int j=0;j<=esl;j++){
                for(int k=0;k<=esl;k++){
                    normdenoms[j]+=alpha[i][k]*transmat[k][j];
                }
            }
            for(int j=0;j<=esl;j++){
                for(int k=0;k<=esl;k++){
                    if(normdenoms[k]==0) gamma[i][j]=0;
                    else{
                        gamma[i][j]+=alpha[i][j]*transmat[j][k]*gamma[i+1][k]/normdenoms[k];
                    }
                }
            }
        }
        return gamma;

    }
    public double[][] learnAlignmente2f(SentencePair sentencePair){
        int esl = sentencePair.getEnglishWords().size();
        int fsl = sentencePair.getFrenchWords().size();

        double[][] alpha=new double[esl][fsl+1];
        double [][]gamma=new double[esl][fsl+1];

        double[][] transmat=new double[fsl+1][fsl+1];

        for(int i=0;i<fsl;i++){
            transmat[i][fsl]=nullposprob;
            transmat[fsl][i]=(1-nullposprob)/fsl;
            double normalization=0;
            for(int j=0;j<fsl;j++){
                transmat[i][j]=Math.exp(-2*Math.abs(j-i-1.1));
                normalization+=transmat[i][j];
            }
            for(int j=0;j<fsl;j++){
                transmat[i][j]=transmat[i][j]*(1-nullposprob)/normalization;
            }
        }
        String firstew=sentencePair.getEnglishWords().get(0);
        int firstei=englishWords.addAndGetIndex(firstew);

        double normalization=0;
        for(int i=0;i<fsl;i++){
            String fw=sentencePair.getFrenchWords().get(i);
            int fi=frenchWords.addAndGetIndex(fw);
            alpha[0][i]= e2fpairCounter.getCount(fi,firstei)*transmat[i][fsl];
            normalization+=alpha[0][i];
        }
        alpha[0][fsl]= e2fpairCounter.getCount(-1,firstei)*transmat[fsl][fsl];
        normalization+=alpha[0][fsl];
        if(normalization!=0){
            for(int i=0;i<fsl;i++) alpha[0][i]=alpha[0][i]/normalization;
        }

        for(int i=1;i<esl;i++){
            String ew=sentencePair.getEnglishWords().get(i);
            int ei=englishWords.addAndGetIndex(ew);

            normalization=0;
            for(int j=0;j<fsl;j++){
                for(int k=0;k<fsl;k++){
                    alpha[i][j]+=alpha[i-1][k]*transmat[k][j];
                }
                String fw=sentencePair.getFrenchWords().get(j);
                int fi=frenchWords.addAndGetIndex(fw);
                alpha[i][j]*= e2fpairCounter.getCount(fi,ei);
                normalization+=alpha[i][j];
            }
            if(normalization>0){
                for(int j=0;j<fsl;j++) alpha[i][j]=alpha[i][j]/normalization;
            }
        }

        System.arraycopy(alpha[esl-1],0,gamma[esl-1],0,fsl);
        double[] normdenoms=new double[fsl+1];

        for(int i=esl-2;i>=0;i--){
            for(int j=0;j<=fsl;j++){
                for(int k=0;k<=fsl;k++){
                    normdenoms[j]+=alpha[i][k]*transmat[k][j];
                }
            }
            for(int j=0;j<=fsl;j++){
                for(int k=0;k<=fsl;k++){
                    if(normdenoms[k]==0) gamma[i][j]=0;
                    else{
                        gamma[i][j]+=alpha[i][j]*transmat[j][k]*gamma[i+1][k]/normdenoms[k];
                    }
                }
            }
        }
        return gamma;

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
