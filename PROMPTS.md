# User Prompts

```text
This project focuses on automate feature engineering using LLMs, in the end it will be submitted to cloudflare ai optional challenge.
Initially we will discuss the requirements of project, steps, phases and the project roadmap. So you will not implement any code yet.

The main requirements from cloudflare are listed bellow:
LLM (recommend using Llama 3.3 on Workers AI), or an external LLM of your choice
Workflow / coordination (recommend using Workflows, Workers or Durable Objects)
User input via chat or voice (recommend using Pages or Realtime)
Memory or state

Use a .md file to present the proposed plan
Note that the current directory already has the template code for the project
```


```text
I am thinking on the user uploading the dataset, with a size limit, the LLM (deepseek api) analyzes the column names or the dataset, we discuss later. Then it suggests feature transformations following a grammar. This grammar will be present in the prompt and it will contain the allowed transformations/operations. The user defines the target variable too. And the llm only sees training set to prevent data leakage
```


```text
Each user should have multiple projects? will that not overload the project? why not keeping a single project (doesnt even show projects pages) for each user and after we have the mvp we add project system?
```


```text
ok, another thing
We need to follow git practices so add to agents.md some rules about good practices to follow using git
Another question is, should i commit and push the current state of the project as the setup?
```


```text
What do you think it should be the next step?
```


```text
update the roadmap by replacing the multiple projects with one active workspace per user/session
The grammar will be later, now i want to focus on the uploading csv mechanism
```


```text
update plan.md with the implementation details plan of the csv upload mechanism
Add there the button to change the theme of the website dark/light
```


```text
I tried upload a avocado.csv and it gaved a error "csv contains empty column names".

Why we dont use python workers? i want to use python
```


```text
I am trying to setup python workers to use why my agent worker but i am having difficulties setup the polars with pywrangler, check that
```


```text
why i cannot keep pandas for cloud worker? i want to keep pandas because on the future i will need the dataset as a dataframe
```


```text
in the future imagine that i want to calculate the transformations made proposed by the llm, what is the best path to chose now
```


```text
in the future imagine that i want to calculate the transformations made proposed by the llm, what is the best path to chose now
```


```text
i want to remove the limit of the max_inference_rows, i want to use the entire dataset for this analysis
```


```text
so, in your vision what would be the best approach and its tradeoffs
```


```text
i liked the chunked approach, so i process by batches so i can fit on memory and at the same time i can do EDA of large datasets
```


```text
i uploaded a file with 2 leading and trailing white spaces, and a recommendation appeared suggesting the change, but clicking on accept or reject doesnt do anything apart from changing the information on the recommendation
```


```text
i uploaded a file with 2 leading and trailing white spaces, and a recommendation appeared suggesting the change, but clicking on accept or reject doesnt do anything apart from changing the information on the recommendation
```


```text
On my current project, after loading the dataset and click to preview the first 20 rows, the box is larger than the above ones. So i want a fixed number of columns in the preview to match the width of the box and then i scroll horizontally to continue to see the rest of the column
```


```text
add the count of unique values to the column dashboard table
Now, on your opinion what should be the next step after this one
```


```text
yes i like that, so the LLM returns the structured assumptions and then user decides to accept or not
The user may want to discuss some changes too so a chat is important
```


```text
I want to use my API key and send a api call for deepseek directly
```


```text
Add agents again to my project, you can check on 6572009bdc3ddc98f5291024689bf06c17d02f2e commit

The objective of the agent is to give insights of the dataset using the metadata and informations of the dataset
```


```text
From the available models of workers AI from cloudflare, which cost effective model should i use for this task?"
```


```text
The reason is not being correctly parsed or the model is not returning is correctly

"The remote model response could not be parsed, so this decision was derived from local profile statistics."
```


```text
This decision system is not good, doesnt say anything
I want to have a simple UI, where the user confirms the proposals
What is your vision about:
- Preview of the first lines of the data is shown
- llm suggests the target variable and the user accepts or denies (if denied select the one that he wants)
- The preprocessing call starts, were the user can call an LLM so suggest the preprocessing steps, like drop because its irrelevant, fill with mean or median and so on
```


```text
its is reasonable to add some samples of each column as context, so new pre processing steps can be made, like splitting name columns

and the decision dropdown i want to contain only the plausible options for each column, so it only shows the proposals of the model, to prevent an option "Fill mean" a binary column
```


```text
After the user accepts the target variable, it changes the state of its column on the column proposals and  the dropbox gets disabled on "use" state
Then i want an option on the end of the column proposal to finish the selection, the columns that the user chosen to drop are dropped and if the user press suggest preprocessing it only uses the columns that are kept
```


```text
After i ask ai to suggest feature transformations this error appeared
Rejected (10)

Rejected

Suggestion does not match the expected AI feature suggestion shape.

Rejected

Suggestion does not match the expected AI feature suggestion shape.

Rejected

Suggestion does not match the expected AI feature suggestion shape.

Rejected

Suggestion does not match the expected AI feature suggestion shape.

Rejected

Suggestion does not match the expected AI feature suggestion shape.

Rejected

Suggestion does not match the expected AI feature suggestion shape.

Rejected

Suggestion does not match the expected AI feature suggestion shape.

Rejected

Suggestion does not match the expected AI feature suggestion shape.

Rejected

Suggestion does not match the expected AI feature suggestion shape.

Rejected

Suggestion does not match the expected AI feature suggestion shape.

Investigate and fix this issue
```


```text
My current feature-generation stage is not fully intelligent
The suggestions are built from the original compact profile and preview rows, scoped only to the kept columns, it doesnt know which preprocessing actions were accepts, what the post preprocessing column types are, or wether preprocessing created/removed/encoded columns.
```


```text
My current pipeline suggest feature engineering, but now its time to materialize those transformations, how do you think this materialization should be done, for example (WeightOverAge)
```


```text
Fix 2 problems:
- the bottom preview dataset is very large, keep the same column as the top preview and enable horizontal scroll
- Agent is not working, i am not able to click or write
```


```text
Its possible to have some templates of csv like titanic or pet adoption dataset, so the user dont need to choose CSV?
```


```text
i pasted the datasets on public, now implement the changes needed to use them
```


```text
Add workspace system on the project, where a user can have multiple workspaces
Note that the agent chat needs to be saved and isolated from each chat, so i the user changes the chat, the agents needs to use the dataset of that workspace and keep the history of the messages on that workspace
```
