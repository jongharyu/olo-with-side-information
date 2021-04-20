# Online learning with side information

## How to load dataset
```python
import datasets
data = datasets.CpuSmall(standardize=False, normalize=True)  # default preprocessing used in (Orabona and Pal, 2016)
data.X  # attributes (data size, dimensions) 
data.y  # target variables (data size,)
```

## Github workflow (very basic)

```bash
# Clone a repository
git clone https://github.com/jongharyu/olo-with-side-information.git

# To create a branch and checkout
git checkout -b branch-name

# How to "update" local changes 
git add file1 file2  # you first need to `add` files to be updated
git commit -m "some meaningful messages"  # then you `commit` those added files with commit message

# Or, if you would add all changed files and commit, you can
git commit -am "some meaningful messages"

# How to push local commits to the remote github repository 
git push origin
```