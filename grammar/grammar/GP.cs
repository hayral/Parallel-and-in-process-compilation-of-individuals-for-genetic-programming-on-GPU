using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;

namespace grammar
{
    public class GP
    {

        public List<Individual> population = new List<Individual>();
        public double totalFitness = 0;
        public double avgFitness = 0;

        public int populationSize = 0;
        public static int genomeSize = 0;
        
        public static Random rnd = new Random();

        public bool minimizeFitness = true;

        public void createPopulation(int populationSize, int GenomeSize)
        {
            this.populationSize = populationSize;
            genomeSize = GenomeSize;

            population.Clear();

            for (int x = 0; x < populationSize; x++)
            {
                population.Add(new Individual()); // as parameter: genome size? genome?
            }
        }




        // ... -> CompileGenotypestoCode -> Evaluate -> Select -> XO -> Mutate -> ...

        public Individual getElite()
        {
            Individual elite = population[0];
            double bestFitness = population[0].fitness;
            if (minimizeFitness)
            {
                for (int x = 1; x < population.Count; x++)
                    if (population[x].fitness < bestFitness)
                    {
                        elite = population[x];
                        bestFitness = elite.fitness;
                    }

            }
            else
            {
                for (int x = 1; x < population.Count; x++)
                    if (population[x].fitness > bestFitness)
                    {
                        elite = population[x];
                        bestFitness = elite.fitness;
                    }
            }
            return elite;
        }

        public double getBestFitness()
        {
            if (minimizeFitness)
                return population.Select<Individual, double>(i => i.fitness).Min();
            else
                return population.Select<Individual, double>(i => i.fitness).Max();
        }

        public List<Individual> tournamentSelection2(bool elitism = true)
        {
            List<Individual> newPopulation = new List<Individual>();
            if (elitism)
            {
                newPopulation.Add(getElite()); 
            }


            for (int x = 0; x < populationSize - (elitism ? 1 : 0); x++)
            {
                Individual aa = population[rnd.Next(populationSize)];
                Individual bb = population[rnd.Next(populationSize)];
                if (minimizeFitness)
                    aa = aa.fitness < bb.fitness ? aa : bb;
                else
                    aa = aa.fitness > bb.fitness ? aa : bb;

                newPopulation.Add(newPopulation.Contains(aa) ? aa.clone() : aa);
            }

            return newPopulation; // you must assign this back to the population explicitly! 
        }

        public List<Individual> tournamentSelection3(bool elitism = true)
        {
            List<Individual> newPopulation = new List<Individual>();
            if (elitism)
                newPopulation.Add(getElite());
           
            for (int x = 0; x < populationSize - (elitism ? 1 : 0); x++)
            {
                Individual aa = population[rnd.Next(populationSize)];
                Individual bb = population[rnd.Next(populationSize)];
                Individual cc = population[rnd.Next(populationSize)];
                if (minimizeFitness)
                {
                    aa = aa.fitness < bb.fitness ? aa : bb;
                    aa = aa.fitness < cc.fitness ? aa : cc;
                }
                else
                {
                    aa = aa.fitness > bb.fitness ? aa : bb;
                    aa = aa.fitness > cc.fitness ? aa : cc;
                }
                
                newPopulation.Add(newPopulation.Contains(aa) ? aa.clone() : aa);
            }

            return newPopulation;
        }


        public void mutate(double rate,bool elitism = true)
        {
            // mutate creates a new individual if mutated or uses same instance if not mutated
            for (int x = (elitism ? 1 : 0); x < populationSize; x++)
            {
                if (rnd.NextDouble() < rate)
                {
                    Individual i = population[x];
                    
                    i.genotype[rnd.Next(i.genomeUsed)] = (uint)rnd.Next(2147483647);
                    
                    i.fitness = -1;  //  always reset fitness if there is genome change
                    
                }
            }

        }

        public void crossover(double rate, bool elitism = true)
        {
            List<Individual> newPopulation = new List<Individual>();
            if (elitism)
            {
                //Individual elite = getElite();
                newPopulation.Add(population[0]);
                population.RemoveAt(0);
                System.Diagnostics.Debug.Assert(population.Count == populationSize - 1);
            }

            uint[] tempGenotype = new uint[genomeSize];

            while (population.Count > 1)
            {
                if (rnd.NextDouble() < rate)
                {
                    Individual aa = population[0];
                    Individual bb = population[1+rnd.Next(population.Count-1)];  // without +1 it randomly selects the 0th individual again which mess up total number of individuals

                    int xopoint = rnd.Next(1, genomeSize - 1);

                    Buffer.BlockCopy(aa.genotype, xopoint * sizeof(int), tempGenotype, 0, (genomeSize - xopoint) * sizeof(int));  //  temp = ----------[aaaaaaaa]
                    Buffer.BlockCopy(bb.genotype, xopoint * sizeof(int), aa.genotype, xopoint * sizeof(int), (genomeSize-xopoint) * sizeof(int));  //  aa = aaaaaaabbbbbbb
                    Buffer.BlockCopy(tempGenotype, 0, bb.genotype, xopoint * sizeof(int), (genomeSize - xopoint) * sizeof(int));  //  bb = bbbbbbbttttttt


                    //reset fitness
                    aa.fitness = -1;
                    bb.fitness = -1;

                    //move to new population
                    newPopulation.Add(aa);
                    newPopulation.Add(bb);
                    population.Remove(aa);
                    population.Remove(bb);
                }
                else
                {
                    Individual aa = population[0];
                    newPopulation.Add(aa);
                    population.Remove(aa);
                }
            }
            //Ensure the elite to be is on position 0
            System.Diagnostics.Debug.Assert(population.Count <= 1);
            newPopulation.AddRange(population);
            population = newPopulation;
            System.Diagnostics.Debug.Assert(population.Count == populationSize);
            
        }

        public double[] fitnessArray()
        {
            return population.Select<Individual, double>(x => x.fitness).ToArray();
        }

    }
}
