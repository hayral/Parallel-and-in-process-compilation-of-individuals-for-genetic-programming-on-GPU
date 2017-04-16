using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace grammar
{
    public class Individual
    {
        public uint[] genotype;

        public bool elite = false;
        
        public double fitness = -1;
        
        public int pos = -1;
        public int genomeUsed = 0;
        public bool wrapAround = false;

        public string code;
        public string oldcode;

        public static byte[] tempByteArray = new byte[GP.genomeSize * sizeof(int)];

        public Individual(uint[] genotype = null)
        {
            if (genotype != null)
            {
                this.genotype = new uint[GP.genomeSize];
                Buffer.BlockCopy(genotype, 0, this.genotype, 0, GP.genomeSize * sizeof(int)); // faster than Array.Copy which inturn is faster than clone
            }
            else
            {
                // typecast random bytes to int[]  ,  alternatives:  BitConverter, Buffer.BlockCopy, unsafe+fixed(?)
                // benchmark if faster than for+next()
                this.genotype = new uint[GP.genomeSize];
                GP.rnd.NextBytes(tempByteArray);
                Buffer.BlockCopy(tempByteArray, 0, this.genotype, 0, GP.genomeSize * sizeof(int));
            }
        }


        public void compileGenotypeToCode(Grammar grammar)
        {
            pos = -1;  // reset position
            oldcode = code;
            code = grammar.compileGenotypeToCode(this);
            genomeUsed = pos;
        }



        public void compileGenotypeToCodeAndComputeFitness()
        {
            // TODO: merge compile and fitness?
        }

        public Individual clone()  //  update as new fields are added
        {
            Individual ni = new Individual(genotype); 
            ni.pos = pos;
            ni.genomeUsed = genomeUsed;
            ni.wrapAround = wrapAround;
            ni.fitness = fitness;
            ni.code = code;
            return ni;
        }

        public int readValueFromGenotype(int modulus)
        {
            pos++;
            if (pos == genotype.Length)
            {
                pos = 0;
                wrapAround = true;
            }
            return (int)(genotype[pos] % modulus);

        }

        public override string ToString()
        {
            return "Individual, fit: "+ fitness.ToString()+" [" + genotype[0].ToString() + "," + genotype[1].ToString();
        }
    }
}
