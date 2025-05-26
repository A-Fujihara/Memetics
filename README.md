# Memetics

## My Learning Journey

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



