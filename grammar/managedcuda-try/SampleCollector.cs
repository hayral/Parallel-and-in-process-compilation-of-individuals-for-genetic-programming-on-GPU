using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace managedcuda_try
{
    class Row : List<Double>
    {
        public Row()
        {

        }

        public Row(double n1) : base()
        {
            Add(n1);
        }

        public Row(double n1, double n2)
            : base()
        {
            Add(n1);
            Add(n2);
        }

        public Row(double n1, double n2, double n3)
            : base()
        {
            Add(n1);
            Add(n2);
            Add(n3);
        }

        public Row(double n1, double n2, double n3, double n4)
            : base()
        {
            Add(n1);
            Add(n2);
            Add(n3);
            Add(n4);
        }

        public Row(double n1, double n2, double n3, double n4, double n5)
            : base()
        {
            Add(n1);
            Add(n2);
            Add(n3);
            Add(n4);
            Add(n5);
        }

        public override string ToString()
        {
            return String.Join(",", this);
        }

    }

    class Table : List<Row>
    {
        public void SaveAsCsv(string fileName, string headerLine)
        {
            var f = System.IO.File.CreateText(fileName);
            f.WriteLine(headerLine);
            foreach (var row in this)
            {
                f.WriteLine(row.ToString());
            }
            f.Close();
        }
    }

    class SampleCollector
    {
        public static Dictionary<String, List<Double>> Samples = new Dictionary<string, List<double>>();
        public static Dictionary<String, Table> Tables = new Dictionary<string, Table>();

        public static void Collect(String collectionName, Double sample)
        {
            if (!Samples.ContainsKey(collectionName))
            {
                Samples.Add(collectionName, new List<double>());
            }
            Samples[collectionName].Add(sample);
        }

        public static double Average(String collectionName)
        {
            return Samples[collectionName].Average();
        }

        public static double Min(String collectionName)
        {
            return Samples[collectionName].Min();
        }

        public static double Max(String collectionName)
        {
            return Samples[collectionName].Max();
        }

        public static void Reset(String collectionName)
        {
            Samples[collectionName].Clear();
        }



        public static void SaveAsTableRow(string tableName, string collectionName, double n1 = -1, double n2 = -1, double n3 = -1, double n4 = -1)
        {
            var r = new Row();
            if (n1 != -1) r.Add(n1);
            if (n2 != -1) r.Add(n2);
            if (n3 != -1) r.Add(n3);
            if (n4 != -1) r.Add(n4);
            r.AddRange(Samples[collectionName]);
            if (!Tables.ContainsKey(tableName))
            {
                var t = new Table();
                Tables.Add(tableName, t);
            }
            Tables[tableName].Add(r);
        }
    }
}
