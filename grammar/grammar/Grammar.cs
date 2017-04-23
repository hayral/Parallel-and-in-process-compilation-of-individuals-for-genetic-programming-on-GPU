using System.Collections.Generic;
using System;
using System.Collections;
using System.Linq;


namespace grammar
{
    // ----- Grammar
    public abstract class Grammar
    {
        public Token startToken; // root symbol is expected to be assigned on subclasses
        public Dictionary<string, Token> tokenDictionary = new Dictionary<string,Token>();


        
        public abstract string bnf {get;}
        public abstract bool minimizeFitness { get; }

        public abstract string template { get; }
        public abstract string codePreamble { get; }
        public abstract string beforeIndividualTemplate { get; }
        public abstract string afterIndividualTemplate { get; }

        public abstract bool kernelPerIndividual { get; }

        public abstract void SetKernelParameters(ManagedCuda.CudaKernel kernel);
        public abstract object[] KernelParameters();
        public abstract void InitializeGPUBuffers(int POPSIZE);
        public abstract void InitializeHostBuffers();
        public abstract void ReadBackFromGPUBuffersAndComputeFitness(List<Individual> population);
        
        public Dictionary<string, string> defines = new Dictionary<string,string>();
        


        public Grammar()
        {
            if (bnf == "")
            {
                throw new Exception("Grammar subclasses must populate the bnf field!");
            }
            else parseBNF(bnf);
        }



        

        public Token createToken(string name)
        {
            Token t = new Token(name,this);
            tokenDictionary.Add(name, t);
            return t;
        }

        public Token getToken(string name)
        {
            return tokenDictionary[name];
        }

        public Token getOrCreateToken(string name)
        {
            if (tokenDictionary.ContainsKey(name))
            {
                return tokenDictionary[name];
            }
            else return createToken(name);
        }

        

        public TokenSequence expandRecursive(Token t, Individual i){
            TokenSequence codeFragment = new TokenSequence( t.expansionRules[i.readValueFromGenotype(t.expansionRules.Count)]); // shallow copy
            for(int x=0;x<codeFragment.Count;x++)
            {
                if (codeFragment[x] is Token) codeFragment[x] = expandRecursive((Token)codeFragment[x], i);
            }

            return codeFragment;            
        }

        public string compileGenotypeToCode(Individual i)  //  !!  expansion may go to infinity
        {
            string s = String.Join("", expandRecursive(startToken, i));
            return s;
        }


        public void parseBNF(string bnf)
        {
            var lines = bnf.Split(new string[] { "\r\n","\n" }, StringSplitOptions.RemoveEmptyEntries);
            var delimiters = "<>".ToCharArray();            

            var rules = new Dictionary<string, string[]>();
            foreach (string line in lines)
            {
                var splitRule = line.Split(new string[] { "::=" }, StringSplitOptions.None);
                var nonterminalName = splitRule[0].Trim().Trim(delimiters);
                rules.Add(nonterminalName, splitRule[1].Trim().Split("|".ToCharArray(), StringSplitOptions.RemoveEmptyEntries));                
                createToken(nonterminalName);
            }

            var parsedRules = new Dictionary<string, List<TokenSequence>>();            
            foreach (var kv in rules)
            {
                getToken(kv.Key).expansionRules = parseRule(kv.Value);  //  expansion rules in tokens
            }            
        }

        public List<TokenSequence> parseRule(string[] rule)
        {
            var tokenSequences = new List<TokenSequence>();
            TokenSequence ts;
            int a,b;
            foreach (string ss in rule)  //  s | s | s |
            {                
                if (ss.Contains("<")) {
                    string s = ss;
                    ts = new TokenSequence();
                    while (s.Contains("<"))
                    {
                        a = s.IndexOf("<");
                        b = s.IndexOf(">");
                        string leftstring = s.Substring(0, a);
                        if(leftstring.Trim().Length > 0) ts.Add(leftstring);  //  ....<                        
                        ts.Add(getToken(s.Substring(a + 1, b - a - 1)));  //  ....<....>
                        s = s.Remove(0, b + 1);
                    }
                    if (s.Trim().Length > 0) ts.Add(s);

                } else {
                    ts = new TokenSequence(ss);
                }
                tokenSequences.Add(ts);                
            }
            return tokenSequences;
        }

        public override string ToString()
        {
            System.Text.StringBuilder sb = new System.Text.StringBuilder();
            foreach(Token t in tokenDictionary.Values) {
                sb.Append(t.ToString()).Append(" ::= ");
                foreach(var er in t.expansionRules) {
                    sb.Append(er.ToString()).Append(" | ");
                }
                sb.Remove(sb.Length-3,3);
                sb.AppendLine();
            }
            return sb.ToString();
        }


    }


}
