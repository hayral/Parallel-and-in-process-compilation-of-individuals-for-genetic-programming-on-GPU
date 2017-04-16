using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO.MemoryMappedFiles;
using System.Linq;
using System.Text;
using System.Threading;
using System.Threading.Tasks;

namespace managedcuda_try
{
    public class CompileProcess
    {

        public EventWaitHandle evt;
        public EventWaitHandle evt2;
        public MemoryMappedFile mmf;
        public MemoryMappedViewStream stream;
        public Process process = null;
        public int ID;
        public bool available = true;



        public static int count = 0;
        public static List<CompileProcess> pool = new List<CompileProcess>();

        public CompileProcess()
        {
            ID = ++count;
            evt = new EventWaitHandle(false, EventResetMode.AutoReset, "compileprocessevent" + ID.ToString());
            evt2 = new EventWaitHandle(false, EventResetMode.AutoReset, "compileprocessevent2" + ID.ToString());

            mmf = MemoryMappedFile.CreateNew("compileprocess" + ID.ToString(), (256 + 512) * 1024);   ///  Size of shared mem
            stream = mmf.CreateViewStream();

            process = Process.Start(System.IO.Path.GetDirectoryName(System.Reflection.Assembly.GetExecutingAssembly().Location)+"\\..\\..\\..\\..\\CompileServer\\bin\\x64\\Debug\\CompileServer.exe", ID.ToString());

            process.PriorityClass = ProcessPriorityClass.AboveNormal;
        }

        public void StartCompile(string source)
        {
            var srcBytes = GetBytes(source);
            stream.Position = 0;
            stream.Write(BitConverter.GetBytes(srcBytes.Length), 0, 4);
            stream.Write(srcBytes, 0, srcBytes.Length);

            // 1
            evt.Set();  /// signal daemon process to start compile
            // 4
        }

        

        public byte[] GatherCompileResult()
        {
            stream.Position = 0;

            byte[] arrLengthBytes = new byte[4];
            stream.Read(arrLengthBytes, 0, 4);
            int arrLength = BitConverter.ToInt32(arrLengthBytes, 0);

            byte[] ptx = new byte[arrLength];
            stream.Read(ptx, 0, arrLength);
            return ptx;
        }


        static byte[] GetBytes(string str)
        {
            byte[] bytes = new byte[str.Length * sizeof(char)];
            System.Buffer.BlockCopy(str.ToCharArray(), 0, bytes, 0, bytes.Length);
            return bytes;
        }

        static string GetString(byte[] bytes)
        {
            char[] chars = new char[bytes.Length / sizeof(char)];
            System.Buffer.BlockCopy(bytes, 0, chars, 0, bytes.Length);
            return new string(chars);
        }

        public static CompileProcess GetFromPool()
        {
            foreach(var cp in pool){
                if (cp.available)
                {
                    cp.available = false;
                    return cp;
                }
            }
            CompileProcess newcp = new CompileProcess();
            newcp.available = false;
            pool.Add(newcp);
            return newcp;            
        }

        public void Release()
        {
            available = true;
        }

    }
}
