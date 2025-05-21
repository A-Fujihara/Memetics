# Memetics

## My Learning Journey

### 19 May 2025
My initial step has been to fork this repo: https://github.com/OmarShehata/semantic-embedding-template.git
I have been working on getting this project to run locally on my machine.  I am running into an issue where there seems to be some compatibility problem with my ARM64 machine. In an attempt to resolve this problem I have been attempting to run the project in WSL (Ubuntu). However, there seems to also be some compatibility issues with GPT4All and Linux. I was wanting to avoid emulating in order to run on x64 or x86, as Linux would provide better performance, but it appears that emulation may be the way forward.

I additionally did some research to understand the nuts and bolts of semantic embedding. Essentially, we want to determine how closely related (semantically speaking) two or more words are to each other (i.e. school, book, studying). One way to do this is to use OpenAI's API. This would tell us precisely what we need to know. However, their API is quite expensive. So another option is to use semantic embedding. This creates a vector that stores associated floating point values. If we take cosine of both of the vectors associated with the words we are comparing, we can determine how close the words are semantically speaking. A lower value indicates no relation, while a higher value indicates a relation between the words. Practically speaking this does have some limitations. For example, if we take the words 'cat' and 'bat', the cosine of their vectors would indicate a relation. However, if 'bat' in this context is actually referring to a 'baseball bat' the equivalency would be inaccurate. While this does clearly impose some practical limitations, awareness is key here. Having awareness of this limitation creates a situation where we can foresee this, and correct for it.

https://link.springer.com/article/10.1007/s11063-020-10376-8

https://medium.com/researchify/exploring-cosine-similarity-how-sentence-embedding-models-measure-meaning-1b047675ef8a

