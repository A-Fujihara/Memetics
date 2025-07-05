# Memetics

## My Learning Journey

### 3 July 2025

I built a system to collect tweets, generate semantic embeddings, and visualize them in an interactive 2D cluster map. What seemed straightforward turned into a journey through platform compatibility issues and machine learning quirks.

**Final Architecture:**
```
Tweet Collection → LM Studio Embeddings → UMAP Reduction → Streamlit Visualization
```

## Challenge 1: ARM64 Windows Compatibility

**Problem:** I started development on ARM64 Windows. Tweet collection worked fine, but data science libraries (`pandas`, `numpy`, `umap-learn`) had major compatibility issues.

**Solution:** I migrated the entire project to WSL (Windows Subsystem for Linux). All libraries work perfectly in Ubuntu.

**Lesson:** For data science work, ARM64 Windows still has gaps. WSL is your friend.

## Challenge 2: LM Studio Networking Issues

**Problem:** I needed LM Studio (Windows) to talk to Python scripts (WSL) for embedding generation. Standard `localhost:1234` failed completely.

**Failed attempts:**
- Windows Firewall rules
- Different IP addresses  
- Disabling firewall entirely

**Solution:**
1. Find Windows IP: `cat /etc/resolv.conf` in WSL
2. Enable "Serve on Local Network" in LM Studio
3. Use actual IP: `http://10.0.0.7:1234/v1/embeddings`

Result: Embedding generation went from impossible to under 1 minute for 500+ tweets.

**Lesson:** WSL networking isn't just "Linux on Windows" - it's a separate network environment.

## Challenge 3: The UMAP Consistency Mystery

**Problem:** I added `random_state=42` to UMAP for consistent visualizations, but adding test tweets completely rearranged the entire layout.

**Debugging:** I added logging to verify the random seed was working:
```python
print(f"UMAP random_state: {reducer.random_state}")
print(f"First few coordinates: {embedding_2d[:3]}")
```

**Discovery:** The random seed WAS working perfectly! Same data = identical coordinates every time.

**Reality Check:** UMAP recalculates the entire layout when you add ANY new data points. This is correct behavior - UMAP optimizes the global layout considering all points together.

**Lesson:** Deterministic ≠ stable coordinates. UMAP prioritizes optimal layout over coordinate stability.

## Key Learnings

1. **Platform matters**: Check compatibility early, especially on ARM64
2. **Local AI is powerful**: LM Studio provides speed + privacy that cloud APIs can't match
3. **Debug assumptions**: When algorithms behave unexpectedly, verify what you think you know
4. **User expectations vs. reality**: Sometimes correct behavior feels wrong to users

## Results

✅ Collects tweets from multiple users  
✅ Generates embeddings locally with LM Studio  
✅ Creates interactive 2D visualizations  
✅ Supports test tweet validation  
✅ Runs smoothly on ARM64 Windows via WSL  
✅ Deployable to Streamlit Cloud  

## Conclusion

Building AI-powered data visualization tools is achievable with modern open-source tools, but expect platform quirks and algorithmic surprises. The combination of local AI models, UMAP, and Streamlit creates a compelling stack for text analysis.

**Most important lesson**: Don't fight platform limitations - embrace solutions like WSL that give you the best of both worlds.

---

### 5 June 2025


# UMAP Tweet Embedding Visualization Project Summary

## Project Overview and Initial Setup
I began with the goal of creating a dynamic visualization of tweet embeddings that had already been processed on the Windows side and stored in a CSV file. The embeddings were 768-dimensional vectors representing the semantic content of tweets. My objective was to reduce these high-dimensional embeddings to a 2D visualization that would show clusters of semantically similar tweets, with the requirement that this visualization update dynamically as new data became available.

## Environment Setup and Tool Selection
The project was conducted in a WSL (Windows Subsystem for Linux) environment using VSCode, primarily because Python libraries would not run properly on Windows ARM64 architecture. This compatibility issue forced me to switch to the Linux subsystem to ensure all required libraries could function correctly. I chose to work with Python in a conda environment, which required installing pandas for data manipulation. 

## Data Processing and Technical Challenges
The main technical challenge encountered was handling malformed data in the CSV file. Initially, the script failed due to "nan" values in the embedding column, which caused parsing errors when using Python's `ast.literal_eval()` function. This was resolved by implementing error handling that skipped malformed entries while preserving valid embeddings. The solution involved checking for null values and malformed strings before attempting to parse the embedding arrays.

## Implementation and Execution
The core implementation involved creating a Python script that loaded the CSV file containing 504 tweets with their corresponding 768-dimensional embeddings. The script converted embedding strings to numpy arrays, applied UMAP reduction to transform the data from 768 dimensions to 2 dimensions, and generated a scatter plot visualization. The final output was saved as a PNG file showing the semantic clustering of tweets in 2D space, where similar content appears grouped together visually.

## Results and Dynamic Visualization Requirements
The static approach successfully generated the visualization, processing all 504 valid embeddings and creating a clear scatter plot that revealed semantic clusters within the tweet data. However, this approach only created a snapshot and didn't meet the dynamic updating requirements. Dynamic updates will be the next progression.

## Technical Methodology Summary
The methodology can be summarized as: semantic embeddings were generated and stored in CSV format on the Windows side, then imported into the WSL environment where pandas was used for data loading and manipulation. A Python script leveraging UMAP reduced the dimensionality from 768D to 2D and created scatter plot visualizations.

## Future Direction and Dynamic Implementation
While the static PNG generation proved the concept successfully, the project's ultimate goal requires dynamic updating capabilities. The next phase involves implementing a Streamlit-based web dashboard that can monitor the CSV file for changes and automatically regenerate the UMAP visualization in real-time, providing a continuously updated view of tweet semantic clusters as new data is processed and added to the dataset - with a goal of transitioning the CSV file to Supabase as the next progression.

![image](https://github.com/user-attachments/assets/9eef7bbf-90a6-4901-b5d1-418601bae4a3)



### 3 June 2025

# Initial steps at semantic embedding
Starting with the collection of 37,939 tweets from four users (imitationlearn, maimecat, danielgolliner, and mcd0w). The initial data collection went smoothly, storing tweets with metadata like usernames, timestamps, and full text content in a structured CSV format. My goal was to transform these tweets into high-dimensional semantic vectors that could reveal content patterns and relationships when visualized.

The first major challenge emerged when setting up the embedding generation pipeline. I initially configured LMStudio with the Nomic text embedding model (text-embedding-nomic-embed-text-v1.5), which produces 768-dimensional vectors. However, my first embedding script included a 0.1-second delay between API calls that would have resulted in over 63 minutes of pure waiting time for the full dataset - an estimated 44+ hours total processing time. This delay was intended to be polite to the API but proved to be a massive bottleneck.

I somewhat solved the performance issue by completely removing the delay which reduced the processing time from 44 hours to approximately 1-1.5 hours for the full dataset. However, even this 'optimized' timeline felt too long as I was still only testing a subset that equated to less than 3% of the total data available. I made a strategic decision to focus on the smallest subset of data first, selecting just the two users with the fewest tweets (danielgolliner and mcd0w, totaling 6,496 tweets) to validate the pipeline before scaling up.

Even with the reduced dataset, I encountered another estimation challenge as the actual processing rate was significantly slower than anticipated. Instead of the 50ms per tweet I initially estimated, each embedding was taking 7-8 seconds due to API response time, network latency, and model processing overhead. The reality showed approximately 6-8 tweets per minute, making even my reduced dataset require substantial time. I ultimately processed just the mcd0w user's 510 tweets as our proof of concept.

After roughly 80 minutes of processing, I successfully generated embeddings for 504 out of 510 tweets (98.8% success rate) with only 6 errors. The embeddings are stored as JSON arrays in a single CSV column, with each tweet represented as a 768-dimensional vector. This gave me a solid foundation for the next phase: dimensionality reduction using UMAP to compress these 768-dimensional vectors into 2D coordinates for interactive visualization. 



### May 25 2025

# Understanding the Curse of Dimensionality and the Need for Dimensionality Reduction

I need to understand **dimensionality reduction** because it addresses a fundamental problem in data analysis known as the **curse of dimensionality**. 

When I work with datasets, each feature or attribute represents one dimension - for instance, if I am analyzing customer data, dimensions might include:

- Age
- Income  
- Purchase history
- Location

The challenge arises because as the number of dimensions increases, the mathematical properties of the space change in counterintuitive ways. 

## The Geometric Problem

To understand this, I must consider what happens when I define the "center" and "edge" of my data space:

- **Center**: Where all dimensions have their average or typical values
- **Edges**: Where at least one dimension has an extreme value

As I add more dimensions, the probability that any data point will have average values across *all* dimensions simultaneously becomes vanishingly small, because there are exponentially more ways to be non-average in at least one dimension than to be average in every single dimension. 

This is a **geometric property**: most of the volume in a high-dimensional space exists near its boundaries rather than its center, so data points are statistically more likely to be located there.

## The Distance Problem

Simultaneously, the concept of distance becomes problematic because when I calculate similarity between data points across many dimensions, the distances between any two points converge toward similar values, making it difficult to distinguish between truly similar and dissimilar observations. 

## The Solution

This phenomenon undermines the effectiveness of machine learning algorithms that rely on distance metrics. **Dimensionality reduction techniques** allow me to identify the most informative dimensions while discarding redundant features, thereby preserving meaningful structure while avoiding these computational and statistical problems.

### 25 May 2025

### Resolving "ERR_MODULE_NOT_FOUND: Cannot find package 'gpt4all'" on ARM64 Windows

#### Problem Description

I encountered an `Error [ERR_MODULE_NOT_FOUND]: Cannot find package 'gpt4all'` when running the `pnpm clustering` script within my `semantic-embedding-template` project. This occurred on my ARM64 Windows laptop. The initial symptom was a `node-gyp-build` error ("No native build was found for platform=win32 arch=arm64..."), indicating that `gpt4all` could not be compiled or found as a pre-built binary for my specific ARM64 architecture and Node.js version.

Despite my intention to use LM Studio for all embedding operations, `gpt4all` was listed as a direct dependency in my `package.json`. Further investigation revealed that the `lib/embeddings.js` module, which is crucial for the `clustering` script, was explicitly importing and attempting to use `gpt4all` for embedding generation. Since `gpt4all` doesn't provide native binaries for ARM64 Windows, and I wanted to leverage LM Studio, a direct code modification was necessary to resolve this conflict.

#### Steps Taken to Fix

The solution involved a two-pronged approach: first, eliminating the problematic `gpt4all` dependency from the project's configuration, and second, directly modifying the `lib/embeddings.js` file to integrate LM Studio for embedding generation instead.

1.  **Removed `gpt4all` from `package.json`**:
    * I opened my `package.json` file located at `C:\Users\fujid\Desktop\memetics\semantic-embedding-template\package.json`.
    * I removed the line `"gpt4all": "^4.0.0",` from the `dependencies` section.
    * This step prevented `pnpm` from attempting to install `gpt4all` altogether, which resolved the initial `node-gyp-build` error.

2.  **Modified `lib/embeddings.js` for LM Studio Integration**:
    * I opened `C:\Users\fujid\Desktop\memetics\semantic-embedding-template\lib\embeddings.js`.
    * I replaced its entire content with a revised version that utilizes the LM Studio local server API for embedding generation. Key changes included:
        * Removing the import statement for `gpt4all`: `import { loadModel, createEmbedding } from 'gpt4all'`.
        * Adding configuration constants for the LM Studio API endpoint (`http://localhost:1234/v1/embeddings`) and the specific embedding model running in LM Studio (e.g., `'nomic-embed-text'`).
        * Rewriting the `embed` asynchronous method to perform a `POST` request to the LM Studio API, passing the text input and extracting the embedding vector from the API's JSON response. Robust error handling for the API call was also integrated.

3.  **Clean Reinstallation of Dependencies**:
    * After modifying both `package.json` and `lib/embeddings.js`, I performed a clean reinstallation of all project dependencies to ensure that `gpt4all` was fully removed and all other necessary packages were correctly installed.
    * I navigated to my project directory (`C:\Users\fujid\Desktop\memetics\semantic-embedding-template`) in my PowerShell terminal and ran the following commands:

    ```powershell
    Remove-Item -Recurse -Force node_modules
    Remove-Item pnpm-lock.yaml
    pnpm store prune
    pnpm cache clean
    pnpm install
    ```

After these steps, executing `pnpm clustering` successfully ran the script, now utilizing LM Studio for embedding generation as intended, completely bypassing the problematic `gpt4all` dependency.


### 23 May 2025

I spent a sizeable amount of time attempting to get GPT4All working locally - initially WSL using Ubuntu before migrating back to Windows 11. Ultimately, I did not have any luck. Everytime I would run the command:

**pnpm simple-embedding**

it resulted in the following error:

**ELIFECYCLE  Command failed with exit code 1.**

I tried multiple LLMs, to no avail. 


While still in WSL:

I was also having difficulties with the project recognizing a Vectra installation. To install Vectra, I ran:

**npm install vectra**

followed by:

**sudo systemctl status vectra**

which repeatedly resulted in:

![image](https://github.com/user-attachments/assets/7734cc8e-972c-496a-a6aa-9abad799a2a1)

By this point I had invested some serious time into getting GPT4All to work and wanted to avoid becoming frustrated and see it through. However, there comes a point where I think we need to accept that maybe the result is simply going to be, 'not today'.

Switching to LMStudio was actually pretty straightforward. I went to the provided link, and conveniently LMStudio had an installer for ARM64 and I downloaded it. Next I downloaded the recommended model: **https://huggingface.co/nomic-ai/nomic-embed-text-v1.5-GGUF**

I fired up LMStudio, and turned the green toggle on to start the server

![image](https://github.com/user-attachments/assets/d29f5064-e526-4adc-af4b-34ddc0693184)

and ran the following pnpm command in the VSCode terminal:

**pnpm lmstudio-embedding**

it then generated the following output in the terminal indicating success:

![image](https://github.com/user-attachments/assets/c2e32d2f-3b47-4ed2-8be4-e197a395e219)

I currently am still troubleshooting why 

**pnpm clustering** 

is resulting in the same

**ELIFECYCLE  Command failed with exit code 1.**


### 19 May 2025
My initial step has been to fork this repo: https://github.com/OmarShehata/semantic-embedding-template.git
I have been working on getting this project to run locally on my machine.  I am running into an issue where there seems to be some compatibility problem with my ARM64 machine. In an attempt to resolve this problem I have been attempting to run the project in WSL (Ubuntu). However, there seems to also be some compatibility issues with GPT4All and Linux. I was wanting to avoid emulating in order to run on x64 or x86, as Linux would provide better performance, but it appears that emulation may be the way forward.

I additionally did some research to understand the nuts and bolts of semantic embedding. Essentially, we want to determine how closely related (semantically speaking) two or more words are to each other (i.e. school, book, studying). One way to do this is to use OpenAI's API. This would tell us precisely what we need to know. However, their API is quite expensive. So another option is to use semantic embedding. This creates a vector that stores associated floating point values. If we take cosine of both of the vectors associated with the words we are comparing, we can determine how close the words are semantically speaking. A lower value indicates no relation, while a higher value indicates a relation between the words. Practically speaking this does have some limitations. For example, if we take the words 'cat' and 'bat', the cosine of their vectors would indicate a relation. However, if 'bat' in this context is actually referring to a 'baseball bat' the equivalency would be inaccurate. While this does clearly impose some practical limitations, awareness is key here. Having awareness of this limitation creates a situation where we can foresee this, and correct for it.

https://link.springer.com/article/10.1007/s11063-020-10376-8

https://medium.com/researchify/exploring-cosine-similarity-how-sentence-embedding-models-measure-meaning-1b047675ef8a



