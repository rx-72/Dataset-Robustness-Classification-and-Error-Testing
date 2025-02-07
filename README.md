# Goal:
Using ZORRO as test measure to determine the robustness of a dataset in response to worst case data uncertainty. By fine tuning the error injection and robustness models, we can determine the susceptibility of a dataset towards providing an incorrect representation of the ground truth. The promise of this research is the ability to not only determine a measure of how robust a given dataset is, but also develop a method to reverse engineer error uncertainty in datasets to capture the ground truth as close as possible. This relies heavily on finding good representational patterns for each dataset which we utilize in our pattern mining approach. We'll also showcase comparisons of robustness across different methods to demontrate the validity of our robustness methods and confirmation of which datasets may be more robust than others

### Minimum recommended system requirements: 
8 cpu, 16 GM RAM on dsmlp

### How to run 
1. Clone repository and change directory location to it (git clone -> cd)
   
2. Run "conda env create -f environment.yml" -> "conda activate Robustness_test_dependencies"
   
3. run  python run.py --dataset {(uploaded datasets)} --task {(chosen task)}"
   

      dataset: {"mpg", "ins", "bos", "fire", "all"}

      task: {'Pattern_Mining', 'Pattern_Testing_Flawed', 'Pattern_Testing_Set_Percent', 'Normalization', 'leave_one_out']}


   The "Leave one out" and "Pattern Testing Flawed" tests will print a heat map showcasing robustness ratio deterioration utilizing a specific chosen model and metric. Multiple heatmaps will be generated per hyperparameter combination including one heatmap based on a naive random indice error injection on Meyer and Zorro. Note the Leave one out method is inferior to the the pattern testing method for the most part, it is simply included as a baseline method and a way to compare results. Also note that the Pattern Testing Flawed is named that way since the method of target indices that are generated is flawed with respect to percentage. The only unflawed result are the 10% column sections of the dataset, are other columns rely on grabbing indices that are not othered in a way of importance (since pattern mining expects equals importance across all indices it captures) The both methods can be ran on all four datasets ('mpg', 'ins', 'bos', 'fire') but cannot be ran on with 'all'. You'll need to declare what test you want to run: (Pattern_Testing_Flawed or leave_one_out) and what dataset to run it upon: ('mpg', 'ins', 'bos', 'fire'). Note there's no baseline test since it's automatically generated as a heatmap as well for comparison on both test, being called the "Naive Method".


### References:
https://gopher-sys.github.io/index.html#papers - Gopher source implementation

https://arxiv.org/pdf/2405.18549 - ZORRO Basis and code reference paper




