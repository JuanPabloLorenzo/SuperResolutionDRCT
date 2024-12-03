## Unofficial Implementation of [DRCT: Saving Image Super-Resolution away from Information Bottleneck](https://arxiv.org/pdf/2404.00722).

This repository provides an unofficial implementation of the DRCT architecture for image super-resolution, as described in the research paper.

### Motivation
After reading the paper, I decided to dive deeper by implementing it by myself.
While exploring the [official codebase](https://github.com/ming053l/DRCT), I noticed some [inconsistencies](https://github.com/ming053l/DRCT/issues/25) between the paper and the code.
This motivated me to modify the official code to better align with the paper and to gain a clearer understanding of the methodology.
Keep in mind that the primary goal of this this implementation is to learn and experiment with the DRCT architecture, so you will probably find restructured code that does the same
thing as the original implementation but is reorganized.

### Recent Updates
While working on this implementation, I noticed that a new version of the paper was recently uploaded. This update addresses some of the inconsistencies I had identified,
such as removing the Swin-Dense-Residual-Connected Block (SDRCB) that don't exists in the official codebase but did exists on the prevous version of the paper.
I will remove SDRCB from the code in line with these updates. They still have some unsued functions and classes copied from
other repositories like the [HAT](https://github.com/XPixelGroup/HAT) or [Swin Transformer](https://github.com/XPixelGroup/HAT) repos, so my implementation cleans up
unnecessary code to make it more concise and focused.

### Current status
This project is a work in progress. While the code currently works as intended and produces good results, I plan to continue refining and improving it to enhance clarity and maintainability.
Future updates will reflect changes from the latest paper revision and further optimizations.

