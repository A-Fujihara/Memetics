# Memetics

## My Learning Journey

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



