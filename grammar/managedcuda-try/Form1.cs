using System;
using System.Collections.Generic;
using System.Text;
using System.Windows.Forms;

using grammar;
using ManagedCuda;
using System.Diagnostics;

using System.Threading;
using System.Threading.Tasks;
using ManagedCuda.BasicTypes;
using System.Linq;
using ManagedCuda.NVRTC;
using System.Runtime.InteropServices;
using System.IO;

using Cudafy;
using Cudafy.Compilers;

namespace managedcuda_try
{
    public partial class Form1 : Form
    {
        public Form1()
        {
            InitializeComponent();
            Environment.SetEnvironmentVariable("CUDA_​CACHE_​DISABLE", "1");  //  *** Optimization?
            var p = Process.GetCurrentProcess();
            p.PriorityClass = ProcessPriorityClass.AboveNormal;
            comboBox1.SelectedIndex = 2;
        }


        const int POPSIZE = 16;
        const int GENCOUNT = 100;
        const int NUMTESTCASE = 128;

        const int PARALLELISM_LEVEL = 2;
        const int MAX_TESTCASE_LEN = 8;
        const int GENOME_SIZE = 13*17;
        
        const float XORATE = 0.7F;
        const float MUTATION_RATE = 0.7F;


   

        private void comboBox1_SelectedIndexChanged(object sender, EventArgs e)
        {
            parallelismUpDown.Enabled = (comboBox1.SelectedIndex != 2);
        }

        private void popSizeMaxUpDown_ValueChanged(object sender, EventArgs e)
        {
            popSizeStepUpDown.Enabled = (popSizeMaxUpDown.Value > popSizeUpDown.Value);
        }

        private void Search_Problem_Button_Click(object sender, EventArgs e)
        {
                    
            var exp = new ExperimentV2();
            exp.GENCOUNT = (int)genCountUpDown.Value;
            exp.NUMTESTCASE = (int)testCaseUpDown.Value;
            exp.PARALLELISM_LEVEL = (int)parallelismUpDown.Value;
            exp.GENOME_SIZE = GENOME_SIZE;
            exp.XORATE = XORATE;
            exp.MUTATION_RATE = MUTATION_RATE;


            for (int pop = (int)popSizeUpDown.Value; pop <= popSizeMaxUpDown.Value; pop += (int)popSizeStepUpDown.Value)
            {
                exp.POPSIZE = pop;                

                for (int rep = 0; rep < repeatUpDown.Value; rep++)
                {
                    textBox1.Text = string.Format("pop: {0} , rep: {1}", pop, rep);  
                    Application.DoEvents();

                    switch (comboBox1.SelectedIndex)  
                    {
                        case 0: // Multiple NVCC launch
                            exp.RunExperiment(new SearchGrammar(exp.NUMTESTCASE, 8), false, false);
                            break;
                        case 1: // Multiple remote NVRTC
                            exp.RunExperiment(new SearchGrammar(exp.NUMTESTCASE, 8), true,true);
                            break;

                        case 2: // Single thread, NVRTC
                            exp.PARALLELISM_LEVEL = 1;
                            exp.RunExperiment(new SearchGrammar(exp.NUMTESTCASE, 8), true, false);
                            break;
                    }

                    

                    SampleCollector.SaveAsTableRow("data", "other", comboBox1.SelectedIndex, exp.POPSIZE, 1, (double)parallelismUpDown.Value);
                    SampleCollector.SaveAsTableRow("data", "JIT", comboBox1.SelectedIndex, exp.POPSIZE, 3, (double)parallelismUpDown.Value);
                    SampleCollector.SaveAsTableRow("data", "PTX", comboBox1.SelectedIndex, exp.POPSIZE, 2, (double)parallelismUpDown.Value);
                    SampleCollector.Reset("other");
                    SampleCollector.Reset("JIT");
                    SampleCollector.Reset("PTX");
                }
            }
            SampleCollector.Tables["data"].SaveAsCsv("data-search.csv", "CompileType,PopulationSize,Total1Ptx2Jit3,ParallelismLevel");
        }

        private void Keijzer6_Button_Click(object sender, EventArgs e)
        {
            testCaseUpDown.Value = 50;

            var exp = new ExperimentV2();
            exp.GENCOUNT = (int)genCountUpDown.Value;
            exp.NUMTESTCASE = (int)testCaseUpDown.Value;
            exp.PARALLELISM_LEVEL = (int)parallelismUpDown.Value;
            exp.GENOME_SIZE = GENOME_SIZE;
            exp.XORATE = XORATE;
            exp.MUTATION_RATE = MUTATION_RATE;


            for (int pop = (int)popSizeUpDown.Value; pop <= popSizeMaxUpDown.Value; pop += (int)popSizeStepUpDown.Value)
            {
                exp.POPSIZE = pop;

                for (int rep = 0; rep < repeatUpDown.Value; rep++)
                {
                    textBox1.Text = string.Format("pop: {0} , rep: {1}", pop, rep);  
                    Application.DoEvents();

                    switch (comboBox1.SelectedIndex)  
                    {
                        case 0: // Multiple NVCC launch
                            exp.RunExperiment(new K6Grammar(exp.NUMTESTCASE), false, false);
                            break;
                        case 1: // Multiple remote NVRTC
                            exp.RunExperiment(new K6Grammar(exp.NUMTESTCASE), true, true);
                            break;

                        case 2: // Single thread, NVRTC
                            exp.PARALLELISM_LEVEL = 1;
                            exp.RunExperiment(new K6Grammar(exp.NUMTESTCASE), true, false);
                            break;
                    }



                    SampleCollector.SaveAsTableRow("data", "other", comboBox1.SelectedIndex, exp.POPSIZE, 1, (double)parallelismUpDown.Value);
                    SampleCollector.SaveAsTableRow("data", "JIT", comboBox1.SelectedIndex, exp.POPSIZE, 3, (double)parallelismUpDown.Value);
                    SampleCollector.SaveAsTableRow("data", "PTX", comboBox1.SelectedIndex, exp.POPSIZE, 2, (double)parallelismUpDown.Value);
                    SampleCollector.Reset("other");
                    SampleCollector.Reset("JIT");
                    SampleCollector.Reset("PTX");
                }
            }
            SampleCollector.Tables["data"].SaveAsCsv("data-K6.csv", "CompileType,PopulationSize,Total1Ptx2Jit3,ParallelismLevel");
        }

        private void button2_Click(object sender, EventArgs e)
        {
            testCaseUpDown.Value = 1024;

            var exp = new ExperimentV2();
            exp.GENCOUNT = (int)genCountUpDown.Value;
            exp.NUMTESTCASE = (int)testCaseUpDown.Value;
            exp.PARALLELISM_LEVEL = (int)parallelismUpDown.Value;
            exp.GENOME_SIZE = GENOME_SIZE;
            exp.XORATE = XORATE;
            exp.MUTATION_RATE = MUTATION_RATE;


            for (int pop = (int)popSizeUpDown.Value; pop <= popSizeMaxUpDown.Value; pop += (int)popSizeStepUpDown.Value)
            {
                exp.POPSIZE = pop;

                for (int rep = 0; rep < repeatUpDown.Value; rep++)
                {
                    textBox1.Text = string.Format("pop: {0} , rep: {1}", pop, rep);  
                    Application.DoEvents();

                    switch (comboBox1.SelectedIndex)  
                    {
                        case 0: // Multiple NVCC launch
                            exp.RunExperiment(new MulGrammar(exp.NUMTESTCASE), false, false);
                            break;
                        case 1: // Multiple remote NVRTC
                            exp.RunExperiment(new MulGrammar(exp.NUMTESTCASE), true, true);
                            break;

                        case 2: // Single thread, NVRTC
                            exp.PARALLELISM_LEVEL = 1;
                            exp.RunExperiment(new MulGrammar(exp.NUMTESTCASE), true, false);
                            break;
                    }



                    SampleCollector.SaveAsTableRow("data", "other", comboBox1.SelectedIndex, exp.POPSIZE, 1, (double)parallelismUpDown.Value);
                    SampleCollector.SaveAsTableRow("data", "JIT", comboBox1.SelectedIndex, exp.POPSIZE, 3, (double)parallelismUpDown.Value);
                    SampleCollector.SaveAsTableRow("data", "PTX", comboBox1.SelectedIndex, exp.POPSIZE, 2, (double)parallelismUpDown.Value);
                    SampleCollector.SaveAsTableRow("data", "fitness", comboBox1.SelectedIndex, exp.POPSIZE, 4, (double)parallelismUpDown.Value);
                    SampleCollector.Reset("other");
                    SampleCollector.Reset("JIT");
                    SampleCollector.Reset("PTX");
                    SampleCollector.Reset("fitness");
                }
            }
            SampleCollector.Tables["data"].SaveAsCsv("data-MUL.csv", "CompileType,PopulationSize,Total1Ptx2Jit3Fitness4,ParallelismLevel");
        }

        private void HelpButton_Click(object sender, EventArgs e)
        {
            AboutBox1 a = new AboutBox1();
            a.ShowDialog();
        }


    }
}
