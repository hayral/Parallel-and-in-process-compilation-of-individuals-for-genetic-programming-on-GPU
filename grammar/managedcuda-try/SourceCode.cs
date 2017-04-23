using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

using grammar;
using System.Threading;
using System.IO.MemoryMappedFiles;

using System.Diagnostics;
using ManagedCuda.BasicTypes;

namespace managedcuda_try
{
    public class SourceCode
    {


        public string rendered;
        public CompileProcess compileProcess;
        public byte[] ptx = null;
        public byte[] cubin = null;
        public CUmodule mod;

        public int ID;
        public int numberOfIndividuals;

        public SourceCode(int ID = -1)
        {
            this.ID = ID;
        }


        public Dictionary<string, string> defines = new Dictionary<string, string>(){
            {"#lesser#", "<"},
            {"#greater#", ">"},
            {"#nl#","\r\n"},
            {"#or#","|"}
        };


        private StringBuilder mainCodeBlock = new StringBuilder();



        public string render(/*List<Individual>*/ IEnumerable<Individual> individuals,Grammar grm,int startingIndividualID = 0 )
        {
            
            rendered = grm.template;
            rendered = rendered.Replace("#codepreamble#", grm.codePreamble);
            

            mainCodeBlock.Clear();
            int individualId = startingIndividualID;
            foreach(var i in individuals){
                if (i.elite)
                {
                    string code = i.code;
                    i.compileGenotypeToCode(grm);
                    if (code != i.code)
                    {
                        System.Diagnostics.Debug.WriteLine("****");
                    }
                }
                else i.compileGenotypeToCode(grm);

                mainCodeBlock.Append(grm.beforeIndividualTemplate.Replace("#individualId#", individualId.ToString()));
                mainCodeBlock.Append(i.code);
                mainCodeBlock.Append(grm.afterIndividualTemplate.Replace("#individualId#", individualId.ToString()));
                individualId++;
            }
            numberOfIndividuals = individualId-startingIndividualID;

            rendered = rendered.Replace("#codeblock#",  mainCodeBlock.ToString());
            rendered = postProcess(rendered);
            return rendered;
        }



        public string postProcess(string s)
        {
            string ss = s;
            foreach (var d in defines)
            {
                s = s.Replace(d.Key, d.Value);
            }
            return s;
        }

        public void define(string a, string b)
        {
            defines.Add(a, b);
        }





        public void StartCompile()
        {
            compileProcess = CompileProcess.GetFromPool();
            compileProcess.StartCompile(rendered);
        }

        public void Wait()
        {
            compileProcess.evt2.WaitOne();
        }

        public void GatherCompileResult()
        {
            ptx = compileProcess.GatherCompileResult();
            compileProcess.Release();
        }
    }
}

