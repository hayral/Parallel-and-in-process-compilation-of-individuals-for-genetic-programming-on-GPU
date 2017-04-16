namespace managedcuda_try
{
    partial class Form1
    {
        /// <summary>
        /// Required designer variable.
        /// </summary>
        private System.ComponentModel.IContainer components = null;

        /// <summary>
        /// Clean up any resources being used.
        /// </summary>
        /// <param name="disposing">true if managed resources should be disposed; otherwise, false.</param>
        protected override void Dispose(bool disposing)
        {
            if (disposing && (components != null))
            {
                components.Dispose();
            }
            base.Dispose(disposing);
        }

        #region Windows Form Designer generated code

        /// <summary>
        /// Required method for Designer support - do not modify
        /// the contents of this method with the code editor.
        /// </summary>
        private void InitializeComponent()
        {
            this.textBox1 = new System.Windows.Forms.TextBox();
            this.comboBox1 = new System.Windows.Forms.ComboBox();
            this.parallelismUpDown = new System.Windows.Forms.NumericUpDown();
            this.label1 = new System.Windows.Forms.Label();
            this.label2 = new System.Windows.Forms.Label();
            this.popSizeUpDown = new System.Windows.Forms.NumericUpDown();
            this.label3 = new System.Windows.Forms.Label();
            this.testCaseUpDown = new System.Windows.Forms.NumericUpDown();
            this.label4 = new System.Windows.Forms.Label();
            this.genCountUpDown = new System.Windows.Forms.NumericUpDown();
            this.label5 = new System.Windows.Forms.Label();
            this.popSizeMaxUpDown = new System.Windows.Forms.NumericUpDown();
            this.label6 = new System.Windows.Forms.Label();
            this.repeatUpDown = new System.Windows.Forms.NumericUpDown();
            this.label7 = new System.Windows.Forms.Label();
            this.popSizeStepUpDown = new System.Windows.Forms.NumericUpDown();
            this.button4 = new System.Windows.Forms.Button();
            this.button1 = new System.Windows.Forms.Button();
            this.MULbutton = new System.Windows.Forms.Button();
            this.HelpButton = new System.Windows.Forms.Button();
            ((System.ComponentModel.ISupportInitialize)(this.parallelismUpDown)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.popSizeUpDown)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.testCaseUpDown)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.genCountUpDown)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.popSizeMaxUpDown)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.repeatUpDown)).BeginInit();
            ((System.ComponentModel.ISupportInitialize)(this.popSizeStepUpDown)).BeginInit();
            this.SuspendLayout();
            // 
            // textBox1
            // 
            this.textBox1.Anchor = ((System.Windows.Forms.AnchorStyles)((((System.Windows.Forms.AnchorStyles.Top | System.Windows.Forms.AnchorStyles.Bottom) 
            | System.Windows.Forms.AnchorStyles.Left) 
            | System.Windows.Forms.AnchorStyles.Right)));
            this.textBox1.Location = new System.Drawing.Point(12, 119);
            this.textBox1.Multiline = true;
            this.textBox1.Name = "textBox1";
            this.textBox1.Size = new System.Drawing.Size(443, 215);
            this.textBox1.TabIndex = 1;
            // 
            // comboBox1
            // 
            this.comboBox1.DropDownStyle = System.Windows.Forms.ComboBoxStyle.DropDownList;
            this.comboBox1.FormattingEnabled = true;
            this.comboBox1.Items.AddRange(new object[] {
            "NVCC, multiple NVCC.exe launch",
            "NVRTC, multiple resident remote process with mem mapped IPC",
            "NVRTC, local single thread"});
            this.comboBox1.Location = new System.Drawing.Point(12, 12);
            this.comboBox1.Name = "comboBox1";
            this.comboBox1.Size = new System.Drawing.Size(443, 21);
            this.comboBox1.TabIndex = 5;
            this.comboBox1.SelectedIndexChanged += new System.EventHandler(this.comboBox1_SelectedIndexChanged);
            // 
            // parallelismUpDown
            // 
            this.parallelismUpDown.Enabled = false;
            this.parallelismUpDown.Location = new System.Drawing.Point(74, 39);
            this.parallelismUpDown.Maximum = new decimal(new int[] {
            8,
            0,
            0,
            0});
            this.parallelismUpDown.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.parallelismUpDown.Name = "parallelismUpDown";
            this.parallelismUpDown.Size = new System.Drawing.Size(31, 20);
            this.parallelismUpDown.TabIndex = 6;
            this.parallelismUpDown.Value = new decimal(new int[] {
            2,
            0,
            0,
            0});
            // 
            // label1
            // 
            this.label1.AutoSize = true;
            this.label1.Location = new System.Drawing.Point(12, 41);
            this.label1.Name = "label1";
            this.label1.Size = new System.Drawing.Size(56, 13);
            this.label1.TabIndex = 7;
            this.label1.Text = "Parallelism";
            // 
            // label2
            // 
            this.label2.AutoSize = true;
            this.label2.Location = new System.Drawing.Point(167, 41);
            this.label2.Name = "label2";
            this.label2.Size = new System.Drawing.Size(49, 13);
            this.label2.TabIndex = 9;
            this.label2.Text = "Pop Size";
            // 
            // popSizeUpDown
            // 
            this.popSizeUpDown.Increment = new decimal(new int[] {
            20,
            0,
            0,
            0});
            this.popSizeUpDown.Location = new System.Drawing.Point(222, 39);
            this.popSizeUpDown.Maximum = new decimal(new int[] {
            1024,
            0,
            0,
            0});
            this.popSizeUpDown.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.popSizeUpDown.Name = "popSizeUpDown";
            this.popSizeUpDown.Size = new System.Drawing.Size(38, 20);
            this.popSizeUpDown.TabIndex = 8;
            this.popSizeUpDown.Value = new decimal(new int[] {
            40,
            0,
            0,
            0});
            // 
            // label3
            // 
            this.label3.AutoSize = true;
            this.label3.Location = new System.Drawing.Point(305, 41);
            this.label3.Name = "label3";
            this.label3.Size = new System.Drawing.Size(60, 13);
            this.label3.TabIndex = 11;
            this.label3.Text = "Test Cases";
            // 
            // testCaseUpDown
            // 
            this.testCaseUpDown.Increment = new decimal(new int[] {
            32,
            0,
            0,
            0});
            this.testCaseUpDown.Location = new System.Drawing.Point(371, 39);
            this.testCaseUpDown.Maximum = new decimal(new int[] {
            2048,
            0,
            0,
            0});
            this.testCaseUpDown.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.testCaseUpDown.Name = "testCaseUpDown";
            this.testCaseUpDown.Size = new System.Drawing.Size(43, 20);
            this.testCaseUpDown.TabIndex = 10;
            this.testCaseUpDown.Value = new decimal(new int[] {
            32,
            0,
            0,
            0});
            // 
            // label4
            // 
            this.label4.AutoSize = true;
            this.label4.Location = new System.Drawing.Point(12, 67);
            this.label4.Name = "label4";
            this.label4.Size = new System.Drawing.Size(90, 13);
            this.label4.TabIndex = 15;
            this.label4.Text = "Generation Count";
            // 
            // genCountUpDown
            // 
            this.genCountUpDown.Increment = new decimal(new int[] {
            10,
            0,
            0,
            0});
            this.genCountUpDown.Location = new System.Drawing.Point(108, 65);
            this.genCountUpDown.Maximum = new decimal(new int[] {
            500,
            0,
            0,
            0});
            this.genCountUpDown.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.genCountUpDown.Name = "genCountUpDown";
            this.genCountUpDown.Size = new System.Drawing.Size(40, 20);
            this.genCountUpDown.TabIndex = 14;
            this.genCountUpDown.Value = new decimal(new int[] {
            10,
            0,
            0,
            0});
            // 
            // label5
            // 
            this.label5.AutoSize = true;
            this.label5.Location = new System.Drawing.Point(167, 69);
            this.label5.Name = "label5";
            this.label5.Size = new System.Drawing.Size(72, 13);
            this.label5.TabIndex = 17;
            this.label5.Text = "Pop Size Max";
            // 
            // popSizeMaxUpDown
            // 
            this.popSizeMaxUpDown.Increment = new decimal(new int[] {
            20,
            0,
            0,
            0});
            this.popSizeMaxUpDown.Location = new System.Drawing.Point(245, 67);
            this.popSizeMaxUpDown.Maximum = new decimal(new int[] {
            1024,
            0,
            0,
            0});
            this.popSizeMaxUpDown.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.popSizeMaxUpDown.Name = "popSizeMaxUpDown";
            this.popSizeMaxUpDown.Size = new System.Drawing.Size(38, 20);
            this.popSizeMaxUpDown.TabIndex = 16;
            this.popSizeMaxUpDown.Value = new decimal(new int[] {
            40,
            0,
            0,
            0});
            this.popSizeMaxUpDown.ValueChanged += new System.EventHandler(this.popSizeMaxUpDown_ValueChanged);
            // 
            // label6
            // 
            this.label6.AutoSize = true;
            this.label6.Location = new System.Drawing.Point(305, 67);
            this.label6.Name = "label6";
            this.label6.Size = new System.Drawing.Size(42, 13);
            this.label6.TabIndex = 19;
            this.label6.Text = "Repeat";
            // 
            // repeatUpDown
            // 
            this.repeatUpDown.Location = new System.Drawing.Point(353, 65);
            this.repeatUpDown.Maximum = new decimal(new int[] {
            20,
            0,
            0,
            0});
            this.repeatUpDown.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.repeatUpDown.Name = "repeatUpDown";
            this.repeatUpDown.Size = new System.Drawing.Size(41, 20);
            this.repeatUpDown.TabIndex = 18;
            this.repeatUpDown.Value = new decimal(new int[] {
            5,
            0,
            0,
            0});
            // 
            // label7
            // 
            this.label7.AutoSize = true;
            this.label7.Location = new System.Drawing.Point(167, 95);
            this.label7.Name = "label7";
            this.label7.Size = new System.Drawing.Size(74, 13);
            this.label7.TabIndex = 21;
            this.label7.Text = "Pop Size Step";
            // 
            // popSizeStepUpDown
            // 
            this.popSizeStepUpDown.Enabled = false;
            this.popSizeStepUpDown.Increment = new decimal(new int[] {
            5,
            0,
            0,
            0});
            this.popSizeStepUpDown.Location = new System.Drawing.Point(245, 93);
            this.popSizeStepUpDown.Maximum = new decimal(new int[] {
            1024,
            0,
            0,
            0});
            this.popSizeStepUpDown.Minimum = new decimal(new int[] {
            1,
            0,
            0,
            0});
            this.popSizeStepUpDown.Name = "popSizeStepUpDown";
            this.popSizeStepUpDown.Size = new System.Drawing.Size(38, 20);
            this.popSizeStepUpDown.TabIndex = 20;
            this.popSizeStepUpDown.Value = new decimal(new int[] {
            20,
            0,
            0,
            0});
            // 
            // button4
            // 
            this.button4.Location = new System.Drawing.Point(461, 12);
            this.button4.Name = "button4";
            this.button4.Size = new System.Drawing.Size(100, 70);
            this.button4.TabIndex = 22;
            this.button4.Text = "Search Problem";
            this.button4.UseVisualStyleBackColor = true;
            this.button4.Click += new System.EventHandler(this.Search_Problem_Button_Click);
            // 
            // button1
            // 
            this.button1.Location = new System.Drawing.Point(461, 87);
            this.button1.Name = "button1";
            this.button1.Size = new System.Drawing.Size(100, 58);
            this.button1.TabIndex = 23;
            this.button1.Text = "Keijzer-6";
            this.button1.UseVisualStyleBackColor = true;
            this.button1.Click += new System.EventHandler(this.Keijzer6_Button_Click);
            // 
            // MULbutton
            // 
            this.MULbutton.Location = new System.Drawing.Point(461, 151);
            this.MULbutton.Name = "MULbutton";
            this.MULbutton.Size = new System.Drawing.Size(100, 55);
            this.MULbutton.TabIndex = 24;
            this.MULbutton.Text = "5bit MUL";
            this.MULbutton.UseVisualStyleBackColor = true;
            this.MULbutton.Click += new System.EventHandler(this.button2_Click);
            // 
            // HelpButton
            // 
            this.HelpButton.Location = new System.Drawing.Point(461, 310);
            this.HelpButton.Name = "HelpButton";
            this.HelpButton.Size = new System.Drawing.Size(100, 23);
            this.HelpButton.TabIndex = 25;
            this.HelpButton.Text = "Help";
            this.HelpButton.UseVisualStyleBackColor = true;
            this.HelpButton.Click += new System.EventHandler(this.HelpButton_Click);
            // 
            // Form1
            // 
            this.AutoScaleDimensions = new System.Drawing.SizeF(6F, 13F);
            this.AutoScaleMode = System.Windows.Forms.AutoScaleMode.Font;
            this.ClientSize = new System.Drawing.Size(573, 345);
            this.Controls.Add(this.HelpButton);
            this.Controls.Add(this.MULbutton);
            this.Controls.Add(this.button1);
            this.Controls.Add(this.button4);
            this.Controls.Add(this.label7);
            this.Controls.Add(this.popSizeStepUpDown);
            this.Controls.Add(this.label6);
            this.Controls.Add(this.repeatUpDown);
            this.Controls.Add(this.label5);
            this.Controls.Add(this.popSizeMaxUpDown);
            this.Controls.Add(this.label4);
            this.Controls.Add(this.genCountUpDown);
            this.Controls.Add(this.label3);
            this.Controls.Add(this.testCaseUpDown);
            this.Controls.Add(this.label2);
            this.Controls.Add(this.popSizeUpDown);
            this.Controls.Add(this.label1);
            this.Controls.Add(this.parallelismUpDown);
            this.Controls.Add(this.comboBox1);
            this.Controls.Add(this.textBox1);
            this.Name = "Form1";
            this.Text = "ar";
            ((System.ComponentModel.ISupportInitialize)(this.parallelismUpDown)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.popSizeUpDown)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.testCaseUpDown)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.genCountUpDown)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.popSizeMaxUpDown)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.repeatUpDown)).EndInit();
            ((System.ComponentModel.ISupportInitialize)(this.popSizeStepUpDown)).EndInit();
            this.ResumeLayout(false);
            this.PerformLayout();

        }

        #endregion

        private System.Windows.Forms.TextBox textBox1;
        private System.Windows.Forms.ComboBox comboBox1;
        private System.Windows.Forms.NumericUpDown parallelismUpDown;
        private System.Windows.Forms.Label label1;
        private System.Windows.Forms.Label label2;
        private System.Windows.Forms.NumericUpDown popSizeUpDown;
        private System.Windows.Forms.Label label3;
        private System.Windows.Forms.NumericUpDown testCaseUpDown;
        private System.Windows.Forms.Label label4;
        private System.Windows.Forms.NumericUpDown genCountUpDown;
        private System.Windows.Forms.Label label5;
        private System.Windows.Forms.NumericUpDown popSizeMaxUpDown;
        private System.Windows.Forms.Label label6;
        private System.Windows.Forms.NumericUpDown repeatUpDown;
        private System.Windows.Forms.Label label7;
        private System.Windows.Forms.NumericUpDown popSizeStepUpDown;
        private System.Windows.Forms.Button button4;
        private System.Windows.Forms.Button button1;
        private System.Windows.Forms.Button MULbutton;
        private System.Windows.Forms.Button HelpButton;
    }
}

