# SeminarDeepLearning

This is the code repository for the webpage hosted at [https://mlai-bonn.github.io/SeminarDeepLearning/](https://mlai-bonn.github.io/SeminarDeepLearning/).


## How to contribute

For each session write up, please create a markdown file in the root of this repository, containing your write up. 
Please keep to the following naming scheme:

    s0X_TOPIC.md

where X should be replaced by your session number and TOPIC should be replaced by your topic.

Please also edit index.md to contain a new line in the table of contents, containing your name, a link to your github profile, and a link to your markdown file.

Once your markdown file is committed, github will automatically compile it to html and publish it on the [seminar homepage](https://mlai-bonn.github.io/SeminarDeepLearning/).
This might take up to one minute (but is usually faster).
You might need to hard-refresh the page in your browser to see the changes.

### Formulas

Your markdown file might contain latex math mode formulas with certain drawbacks. 
To see examples that work, consider the seminar homepage of earlier installments of our seminar:

#### SS 2019
- [Homepage](https://pwelke.github.io/SeminarFromTheoryToAlgorithms/)
- [GitHub Project](https://github.com/pwelke/SeminarFromTheoryToAlgorithms/)

#### WS 2018/2019
- [Homepage](https://pwelke.github.io/SeminarLearningTheory/)
- [GitHub Project](https://github.com/pwelke/SeminarLearningTheory/)


## How to Clone a New Seminar Homepage from this Repository

1. Clone/Copy this repository to a new Repository ``XXX``
2. Edit ``_config.yaml``: Set ``baseurl: /XXX``
3. If not set already, go to settings of your repository and activate 'Github Pages' for repository ``XXX``
4. Remove all ``s0X_TOPIC.md`` files that may be left from this installment
5. Change ``index.md`` to its initial state.

Alternatively to steps 3.-5. you may just use the tag ``CleanStart2020`` as your starting point.


#### Additional Personalization of Template

The hamburger menu on the left is generated automatically. 
The disclaimer and licensing information in the footer of the template is defined and can be changed in ``_layouts/default.html``.
