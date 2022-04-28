use std::collections::HashMap;

use crate::{TensorBlock, Error, BasicBlock};
use crate::{Labels, LabelValue};

mod utils;

mod keys_to_samples;
mod keys_to_properties;

/// A tensor map is the main user-facing struct of this library, and can store
/// any kind of data used in atomistic machine learning.
///
/// A tensor map contains a list of `TensorBlock`s, each one associated with a
/// key in the form of a single `Labels` entry.
///
/// It provides functions to merge blocks together by moving some of these keys
/// to the samples or properties labels of the blocks, transforming the sparse
/// representation of the data to a dense one.
#[derive(Debug)]
pub struct TensorMap {
    keys: Labels,
    blocks: Vec<TensorBlock>,
    // TODO: arbitrary tensor-level metadata? e.g. using `HashMap<String, String>`
}

#[allow(clippy::needless_pass_by_value)]
fn check_labels_names(
    block: &BasicBlock,
    sample_names: &[&str],
    components_names: &[Vec<&str>],
    context: String,
) -> Result<(), Error> {
    if block.samples().names() != sample_names {
        return Err(Error::InvalidParameter(format!(
            "all blocks must have the same sample label names, got [{}] and [{}]{}",
            block.samples().names().join(", "),
            sample_names.join(", "),
            context,
        )));
    }

    if block.components().len() != components_names.len() {
        return Err(Error::InvalidParameter(format!(
            "all blocks must contains the same set of components, the current \
            block has {} components while the first block has {}{}",
            block.components().len(),
            components_names.len(),
            context,
        )));
    }

    for (component_i, component) in block.components().iter().enumerate() {
        if component.names() != components_names[component_i] {
            return Err(Error::InvalidParameter(format!(
                "all blocks must have the same component label names, got [{}] and [{}]{}",
                component.names().join(", "),
                components_names[component_i].join(", "),
                context,
            )));
        }
    }

    Ok(())
}

impl TensorMap {
    /// TODO: doc
    #[allow(clippy::similar_names)]
    pub fn new(keys: Labels, blocks: Vec<TensorBlock>) -> Result<TensorMap, Error> {
        if blocks.len() != keys.count() {
            return Err(Error::InvalidParameter(format!(
                "expected the same number of blocks ({}) as the number of \
                entries in the keys when creating a tensor, got {}",
                keys.count(), blocks.len()
            )))
        }

        if !blocks.is_empty() {
            // make sure all blocks have the same kind of samples, components &
            // properties labels
            let sample_names = blocks[0].values.samples().names();
            let components_names = blocks[0].values.components()
                .iter()
                .map(|c| c.names())
                .collect::<Vec<_>>();
            let properties_names = blocks[0].values.properties().names();

            let gradients_data = blocks[0].gradients().iter()
                .map(|(name, gradient)| {
                    let components_names = gradient.components()
                        .iter()
                        .map(|c| c.names())
                        .collect::<Vec<_>>();
                    (&**name, (gradient.samples().names(), components_names))
                })
                .collect::<HashMap<_, _>>();


            for block in &blocks {
                check_labels_names(&block.values, &sample_names, &components_names, "".into())?;

                if block.values.properties().names() != properties_names {
                    return Err(Error::InvalidParameter(format!(
                        "all blocks must have the same property label names, got [{}] and [{}]",
                        block.values.properties().names().join(", "),
                        properties_names.join(", "),
                    )));
                }

                if block.gradients().len() != gradients_data.len() {
                    return Err(Error::InvalidParameter(
                        "all blocks must contains the same set of gradients".into(),
                    ));
                }

                for (parameter, gradient) in block.gradients() {
                    match gradients_data.get(&**parameter) {
                        None => {
                            return Err(Error::InvalidParameter(format!(
                                "missing gradient with respect to {} in one of the blocks",
                                parameter
                            )));
                        },
                        Some((sample_names, components_names)) => {
                            check_labels_names(
                                gradient,
                                sample_names,
                                components_names,
                                format!(" for gradients with respect to {}", parameter)
                            )?;
                        }
                    }
                }
            }
        }

        Ok(TensorMap {
            keys,
            blocks,
        })
    }

    /// Get the list of blocks in this `TensorMap`
    pub fn blocks(&self) -> &[TensorBlock] {
        &self.blocks
    }

    /// Get the keys defined in this `TensorMap`
    pub fn keys(&self) -> &Labels {
        &self.keys
    }

    /// Get an iterator over the keys and associated block
    pub fn iter(&self) -> impl Iterator<Item=(&[LabelValue], &TensorBlock)> + '_ {
        self.keys.iter().zip(&self.blocks)
    }

    /// Get the list of blocks matching the given selection. The selection must
    /// contains a single entry, defining the requested key. The selection can
    /// contain only a subset of the variables defined in the keys, in which
    /// case there can be multiple matching blocks.
    pub fn blocks_matching(&self, selection: &Labels) -> Result<Vec<&TensorBlock>, Error> {
        let matching = self.find_matching_blocks(selection)?;

        return Ok(matching.into_iter().map(|i| &self.blocks[i]).collect());
    }

    /// Get a reference to the block matching the given selection.
    ///
    /// The selection behaves similarly to `blocks_matching`, with the exception
    /// that this function returns an error if there is more than one matching
    /// block.
    pub fn block(&self, selection: &Labels) -> Result<&TensorBlock, Error> {
        let matching = self.find_matching_blocks(selection)?;
        if matching.len() != 1 {
            let selection_str = selection.names()
                .iter().zip(&selection[0])
                .map(|(name, value)| format!("{} = {}", name, value))
                .collect::<Vec<_>>()
                .join(", ");


            return Err(Error::InvalidParameter(format!(
                "{} blocks matched the selection ({}), expected only one",
                matching.len(), selection_str
            )));
        }

        return Ok(&self.blocks[matching[0]]);
    }

    /// Actual implementation of `blocks_matching` and related functions, this
    /// function finds the matching blocks & return their index in the
    /// `self.blocks` vector.
    fn find_matching_blocks(&self, selection: &Labels) -> Result<Vec<usize>, Error> {
        if selection.size() == 0 {
            return Ok((0..self.blocks().len()).collect());
        }

        if selection.count() != 1 {
            return Err(Error::InvalidParameter(format!(
                "block selection labels must contain a single row, got {}",
                selection.count()
            )));
        }

        let mut variables = Vec::new();
        'outer: for requested in selection.names() {
            for (i, &name) in self.keys.names().iter().enumerate() {
                if requested == name {
                    variables.push(i);
                    continue 'outer;
                }
            }

            return Err(Error::InvalidParameter(format!(
                "'{}' is not part of the keys for this tensor",
                requested
            )));
        }

        let mut matching = Vec::new();
        let selection = selection.iter().next().expect("empty selection");

        for (block_i, labels) in self.keys.iter().enumerate() {
            let mut selected = true;
            for (&requested_i, &value) in variables.iter().zip(selection) {
                if labels[requested_i] != value {
                    selected = false;
                    break;
                }
            }

            if selected {
                matching.push(block_i);
            }
        }

        return Ok(matching);
    }

    /// Move the given variables from the component labels to the property labels
    /// for each block in this `TensorMap`.
    pub fn components_to_properties(&mut self, variables: &[&str]) -> Result<(), Error> {
        // TODO: requested values
        if variables.is_empty() {
            return Ok(());
        }

        for block in &mut self.blocks {
            block.components_to_properties(variables)?;
        }

        Ok(())
    }
}


#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use crate::eqs_array_t;
    use crate::data::TestArray;
    use crate::LabelsBuilder;

    use super::*;

    fn example_labels(name: &str, count: usize) -> Labels {
        let mut labels = LabelsBuilder::new(vec![name]);
        for i in 0..count {
            labels.add(vec![LabelValue::from(i)]);
        }
        return labels.finish();
    }

    #[test]
    fn blocks_validation() {
        let block_1 = TensorBlock::new(
            eqs_array_t::new(Box::new(TestArray::new(vec![1, 1, 1]))),
            example_labels("samples", 1),
            vec![Arc::new(example_labels("components", 1))],
            Arc::new(example_labels("properties", 1)),
        ).unwrap();

        let block_2 = TensorBlock::new(
            eqs_array_t::new(Box::new(TestArray::new(vec![2, 3, 1]))),
            example_labels("samples", 2),
            vec![Arc::new(example_labels("components", 3))],
            Arc::new(example_labels("properties", 1)),
        ).unwrap();

        let result = TensorMap::new(example_labels("keys", 2), vec![block_1, block_2]);
        assert!(result.is_ok());

        /**********************************************************************/
        let block_1 = TensorBlock::new(
            eqs_array_t::new(Box::new(TestArray::new(vec![1, 1]))),
            example_labels("samples", 1),
            vec![],
            Arc::new(example_labels("properties", 1)),
        ).unwrap();

        let block_2 = TensorBlock::new(
            eqs_array_t::new(Box::new(TestArray::new(vec![2, 1]))),
            example_labels("something_else", 2),
            vec![],
            Arc::new(example_labels("properties", 1)),
        ).unwrap();

        let result = TensorMap::new(example_labels("keys", 2), vec![block_1, block_2]);
        assert_eq!(
            result.unwrap_err().to_string(),
            "invalid parameter: all blocks must have the same sample label \
            names, got [something_else] and [samples]"
        );

        /**********************************************************************/
        let block_1 = TensorBlock::new(
            eqs_array_t::new(Box::new(TestArray::new(vec![1, 1, 1]))),
            example_labels("samples", 1),
            vec![Arc::new(example_labels("components", 1))],
            Arc::new(example_labels("properties", 1)),
        ).unwrap();

        let block_2 = TensorBlock::new(
            eqs_array_t::new(Box::new(TestArray::new(vec![2, 1]))),
            example_labels("samples", 2),
            vec![],
            Arc::new(example_labels("properties", 1)),
        ).unwrap();

        let result = TensorMap::new(example_labels("keys", 2), vec![block_1, block_2]);
        assert_eq!(
            result.unwrap_err().to_string(),
            "invalid parameter: all blocks must contains the same set of \
            components, the current block has 0 components while the first \
            block has 1"
        );

        /**********************************************************************/
        let block_1 = TensorBlock::new(
            eqs_array_t::new(Box::new(TestArray::new(vec![1, 1, 1]))),
            example_labels("samples", 1),
            vec![Arc::new(example_labels("components", 1))],
            Arc::new(example_labels("properties", 1)),
        ).unwrap();

        let block_2 = TensorBlock::new(
            eqs_array_t::new(Box::new(TestArray::new(vec![2, 3, 1]))),
            example_labels("samples", 2),
            vec![Arc::new(example_labels("something_else", 3))],
            Arc::new(example_labels("properties", 1)),
        ).unwrap();

        let result = TensorMap::new(example_labels("keys", 2), vec![block_1, block_2]);
        assert_eq!(
            result.unwrap_err().to_string(),
            "invalid parameter: all blocks must have the same component label \
            names, got [something_else] and [components]"
        );

        /**********************************************************************/
        let block_1 = TensorBlock::new(
            eqs_array_t::new(Box::new(TestArray::new(vec![1, 1]))),
            example_labels("samples", 1),
            vec![],
            Arc::new(example_labels("properties", 1)),
        ).unwrap();

        let block_2 = TensorBlock::new(
            eqs_array_t::new(Box::new(TestArray::new(vec![2, 1]))),
            example_labels("samples", 2),
            vec![],
            Arc::new(example_labels("something_else", 1)),
        ).unwrap();

        let result = TensorMap::new(example_labels("keys", 2), vec![block_1, block_2]);
        assert_eq!(
            result.unwrap_err().to_string(),
            "invalid parameter: all blocks must have the same property label \
            names, got [something_else] and [properties]"
        );

        // TODO: check error messages for gradients
    }
}