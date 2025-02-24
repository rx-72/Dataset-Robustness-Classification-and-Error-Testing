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


The "Leave_one_out" and "Pattern_Testing_Flawed" tests will print a heat map showcasing robustness ratio deterioration utilizing a specific chosen model and metric. Multiple heatmaps will be generated per hyperparameter combination including one heatmap based on a naive random indice error injection on Meyer and Zorro. Note the Leave one out method is inferior to the the pattern testing method for the most part, it is simply included as a baseline method and a way to compare results. Also note that the Pattern Testing Flawed is named that way since the method of target indices that are generated is flawed with respect to percentage. (This is because the leave one out method orders its target indices by importance before capturing a given percentage of it. Meanwhile the pattern mining approach does not, only capturing patterns that fall under the specific number of uncertainity intended to inject and thus not having an order of importance on its target indices). The only unflawed result are the 10% column sections of the dataset, which all our patterns were run upon to discover patterns that capture 10% of the dataset. All other columns rely on grabbing indices that are not ordered in a way of importance (since pattern mining expects equals importance across all indices it captures). Both methods can be ran on all four datasets ('mpg', 'ins', 'bos', 'fire') but cannot be ran on with 'all'. You'll need to declare what test you want to run: (Pattern_Testing_Flawed or leave_one_out) and what dataset to run it upon: ('mpg', 'ins', 'bos', 'fire'). 

Pattern_Testing_Set_Percent is similar to the aforementioned tests but it runs on a fixed percentage of uncertainty at 10%. As such, all results are based on lineplots that rely on the uncertainty radius ('x') and the resulting robustness ratio with that radius ('y'). This task takes in all 4 datasets but does not work with 'all', and will return a lineplot showcasing the results. These plots are to mainly showcase the dominance of the pattern mining approaches throw the patterns found per each dataset, demonstrating the overall dominance in each line plot of meeting the robustness diminishment ("y" value) the quickest.

Pattern Mining is a replication of the pattern mining process. This specifically demonstrates our methods to finding the top 3 patterns associated as the worst case error injection scenario for each given dataset. The return result is a printed message showcasing what were the found "important" patterns for each respective dataset inputted. This task takes in all 4 datasets but does not work with 'all'.

Finally the normalization task will take some error injection methods from a single or multiple dataset and order them with a specific robustness normalized score such we can rank between methods on one dataset or robustness across multiple datasets. If working on a single dataset, the given result is the worst error injection method, or rather the best method for causing the lowest robustness value (or the worst case uncertainity) on that specific dataset vs all other methods. If comparing on multiple datasets, the result is printed as the dataset with the worst robustness value along with the dataset with the best robustness value, thus allowing us to confirm out of a set of datasets which is the most robust and which is the least robust (with respect to tasked predictions). This task takes in all 4 datasets with the single dataset method, but if you want to run the multiple, you need to declare 'all'. A brief explanation of how 'all' is handled is that all datasets are tested with their worst case uncertainity method found in running singular dataset tests before the full blown comparison across one another. This way we get the worst case scenario on each dataset and after scaling across the ranges of values for each dataset, whoever's worst case scenario is more/less problematic can be considered the least/most robust dataset out of the compared datasets. For more information, please see our paper report associated with this repo.


### References:
https://gopher-sys.github.io/index.html#papers - Gopher source implementation

https://arxiv.org/pdf/2405.18549 - ZORRO Basis and code reference paper

https://docs.google.com/document/d/1FoiUCxiMsFaBJMHtvNRZXgJ4mQVd7XqtAp5Wr--bVe4/edit?usp=sharing - Previous Quarter Paper, contains references to previous quarter study and methods




